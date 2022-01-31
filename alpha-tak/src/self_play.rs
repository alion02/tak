use std::sync::mpsc::channel;

use rand::random;
use tak::{
    colour::Colour,
    game::{Game, GameResult},
    tile::Tile,
    turn::Turn,
};
use threadpool::ThreadPool;

use crate::{
    example::{Example, IncompleteExample},
    mcts::Node,
    network::Network,
    turn_map::Lut,
};

const SELF_PLAY_GAMES: usize = 1000;
const ROLLOUTS_PER_MOVE: u32 = 200;
const PIT_GAMES: u32 = 200;
const WIN_RATE_THRESHOLD: f64 = 0.55;
const MAX_EXAMPLES: usize = 1_000_000;
const OPENING_PLIES: usize = 6;

const WORKERS: usize = 4;

/// Run multiple games against self.
fn self_play<const N: usize>(network: &Network<N>) -> Vec<Example<N>>
where
    [[Option<Tile>; N]; N]: Default,
    Turn<N>: Lut,
{
    let pool = ThreadPool::new(WORKERS);
    let (tx, rx) = channel();
    for i in 0..SELF_PLAY_GAMES {
        let tx = tx.clone();
        let nn = copy(network); // copying network because Tensor is not thread-safe
                                // run game in its own thread
        pool.execute(move || {
            tx.send(self_play_game(&nn)).unwrap();
            println!("self-play game {i}/{SELF_PLAY_GAMES}");
        });
    }
    // collect examples
    rx.iter().take(SELF_PLAY_GAMES).fold(Vec::new(), |mut a, b| {
        a.extend(b.into_iter());
        a
    })
}

/// Run a single game against self.
fn self_play_game<const N: usize>(network: &Network<N>) -> Vec<Example<N>>
where
    [[Option<Tile>; N]; N]: Default,
    Turn<N>: Lut,
{
    let mut game_examples = Vec::new();
    let mut game = Game::default(); // TODO add komi?
                                    // make random moves for the first few turns to diversify training data
    for _ in 0..OPENING_PLIES {
        game.nth_move(random()).unwrap();
    }

    // initialize MCTS
    let mut node = Node::default();
    while matches!(game.winner(), GameResult::Ongoing) {
        for _ in 0..ROLLOUTS_PER_MOVE {
            node.rollout(game.clone(), network);
        }
        // and incomplete example
        game_examples.push(IncompleteExample {
            game: game.clone(),
            policy: node.improved_policy(),
        });
        // pick a turn and play it
        let turn = node.pick_move(false);
        node = node.play(&turn);
        game.play(turn).unwrap();
    }
    let winner = game.winner();
    // complete examples by filling in game result
    let result = match winner {
        GameResult::Winner(Colour::White) => 1.,
        GameResult::Winner(Colour::Black) => -1.,
        GameResult::Draw => 0.,
        GameResult::Ongoing => unreachable!(),
    };
    game_examples
        .into_iter()
        .enumerate()
        .map(|(ply, ex)| {
            let perspective = if ply % 2 == 0 { result } else { -result };
            ex.complete(perspective)
        })
        .collect()
}

#[derive(Debug)]
struct PitResult {
    wins: u32,
    #[allow(dead_code)]
    draws: u32,
    losses: u32,
}

impl PitResult {
    pub fn win_rate(&self) -> f64 {
        // (self.wins as f64 + self.draws as f64 / 2.) / (self.wins + self.draws +
        // self.losses) as f64
        self.wins as f64 / (self.wins + self.losses) as f64
    }
}

/// Pits two networks against each other.
/// Returns wins, draws, and losses of the match.
fn pit<const N: usize>(new: &Network<N>, old: &Network<N>) -> PitResult
where
    [[Option<Tile>; N]; N]: Default,
    Turn<N>: Lut,
{
    println!("pitting two networks against each other");
    let mut wins = 0;
    let mut draws = 0;
    let mut losses = 0;

    for i in 0..PIT_GAMES {
        let win_all = PitResult {
            wins: wins + PIT_GAMES - i,
            draws,
            losses,
        };
        let lose_all = PitResult {
            wins,
            draws,
            losses: losses + PIT_GAMES - i,
        };
        if win_all.win_rate() < WIN_RATE_THRESHOLD || lose_all.win_rate() > WIN_RATE_THRESHOLD {
            println!("ending early because result is already determined");
            break;
        }

        println!("pit game: {i}/{PIT_GAMES}");
        // TODO add komi?
        let mut game = Game::default();
        game.opening(i as usize / 2).unwrap();
        let my_colour = if i % 2 == 0 { Colour::White } else { Colour::Black };

        let mut my_node = Node::default();
        let mut opp_node = Node::default();
        while matches!(game.winner(), GameResult::Ongoing) {
            // At least one rollout to initialize all the moves in the trees
            my_node.rollout(game.clone(), new);
            opp_node.rollout(game.clone(), old);
            let turn = if game.to_move == my_colour {
                for _ in 0..ROLLOUTS_PER_MOVE {
                    my_node.rollout(game.clone(), new);
                }
                my_node.pick_move(true)
            } else {
                for _ in 0..ROLLOUTS_PER_MOVE {
                    opp_node.rollout(game.clone(), old);
                }
                opp_node.pick_move(true)
            };
            my_node = my_node.play(&turn);
            opp_node = opp_node.play(&turn);
            game.play(turn).unwrap();
        }
        let game_result = game.winner();
        match game_result {
            GameResult::Winner(winner) => {
                if winner == my_colour {
                    wins += 1;
                } else {
                    losses += 1;
                }
            }
            GameResult::Draw => draws += 1,
            GameResult::Ongoing => unreachable!(),
        }
        println!(
            "{:?} as {:?} [{}/{}/{}]\n{}",
            game_result, my_colour, wins, draws, losses, game.board
        );
    }

    PitResult { wins, draws, losses }
}

/// Do self-play and test against previous iteration
/// until an improvement is seen.
pub fn play_until_better<const N: usize>(network: Network<N>, examples: &mut Vec<Example<N>>) -> Network<N>
where
    [[Option<Tile>; N]; N]: Default,
    Turn<N>: Lut,
{
    loop {
        examples.extend(self_play(&network).into_iter());
        if examples.len() > MAX_EXAMPLES {
            examples.reverse();
            examples.truncate(MAX_EXAMPLES);
            examples.reverse();
        }

        let mut new_network = copy(&network);
        new_network.train(examples);
        let results = pit(&new_network, &network);
        println!("{:?}", results);
        if results.win_rate() > WIN_RATE_THRESHOLD {
            return new_network;
        }
    }
}

fn copy<const N: usize>(network: &Network<N>) -> Network<N> {
    // copy network values by file (UGLY)
    let mut dir = std::env::temp_dir();
    dir.push("model");
    network.save(&dir).unwrap();
    Network::<N>::load(&dir).unwrap()
}
