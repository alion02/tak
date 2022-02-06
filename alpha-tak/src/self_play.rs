use std::{
    sync::{mpsc::{channel}, Arc, atomic::{AtomicBool, Ordering}},
    thread::{self},
};

use arrayvec::ArrayVec;
use rand::random;
use tak::{
    colour::Colour,
    game::{Game, GameResult},
    tile::Tile,
    turn::Turn,
};

use rayon::prelude::*;

use crate::{
    agent::{Agent, Batcher},
    example::{Example, IncompleteExample},
    mcts::Node,
    network::Network,
    turn_map::Lut,
};

const SELF_PLAY_GAMES: usize = 2000;
const ROLLOUTS_PER_MOVE: u32 = 1000;
const OPENING_PLIES: usize = 6;

/// Run multiple games against self.
#[allow(dead_code)]
pub fn self_play<const N: usize>(network: &Network<N>) -> Vec<Example<N>>
where
    [[Option<Tile>; N]; N]: Default,
    Turn<N>: Lut,
{
    (0..SELF_PLAY_GAMES).fold(Vec::new(), |mut examples, i| {
        examples.extend(self_play_game(network).into_iter());
        println!("self-play game {i}/{SELF_PLAY_GAMES}");
        examples
    })
}

/// Run multiple games against self concurrently.
pub fn self_play_async<const N: usize>(network: &Network<N>) -> Vec<Example<N>>
where
    [[Option<Tile>; N]; N]: Default,
    Turn<N>: Lut,
{
    const SIMUL_GAMES: usize = 6;
    println!("Starting self-play with {SIMUL_GAMES} simulatenous games");

    // setup agents
    let mut agents = Vec::with_capacity(SIMUL_GAMES);
    let mut receivers: ArrayVec<_, SIMUL_GAMES> = ArrayVec::new();
    let mut senders: ArrayVec<_, SIMUL_GAMES> = ArrayVec::new();
    let (examples_tx, examples_rx) = channel();
    let should_play = Arc::new(AtomicBool::new(true));
    for _ in 0..SIMUL_GAMES {
        let (game_tx, game_rx) = channel();
        let (policy_tx, policy_rx) = channel();
        receivers.push(game_rx);
        senders.push(policy_tx);
        agents.push((Batcher::new(game_tx, policy_rx), examples_tx.clone(), should_play.clone()));
    }
    
    // start game playing thread
    thread::spawn(move || agents.into_par_iter().for_each(|(agent, tx, should_play)| {
        while should_play.load(Ordering::Relaxed) {
            tx.send(self_play_game(&agent)).unwrap()
        }
    }));

    let mut completed_games = 0;
    let mut total_examples = Vec::new();
    let mut communicators: ArrayVec<_, SIMUL_GAMES> = ArrayVec::new();
    let mut batch: ArrayVec<_, SIMUL_GAMES> = ArrayVec::new();

    let mut average_batch_size = 0.;
    let mut n = 0;
    while completed_games < SELF_PLAY_GAMES {
        // collect game states
        for (i, rx) in receivers.iter().enumerate() {
            if let Ok(game) = rx.try_recv() {
                communicators.push(i);
                batch.push(game);
            }
        }

        if !batch.is_empty() {
            average_batch_size = (average_batch_size * n as f32 + batch.len() as f32) / (n + 1) as f32;
            n += 1;

            // run prediction
            let (policies, evals) = network.policy_eval_batch(&batch);

            // send out outputs
            for (&i, r) in communicators
                .iter()
                .zip(policies.into_iter().zip(evals.into_iter()))
            {
                senders[i].send(r).unwrap();
            }

            communicators.clear();
            batch.clear();
        }

        // collect examples
        while let Ok(examples) = examples_rx.try_recv() {
            completed_games += 1;
            println!("self-play game {completed_games}/{SELF_PLAY_GAMES}");
            println!("average batch size {average_batch_size:.3} with n={n}");
            total_examples.extend(examples.into_iter());
        }

        // stop starting new games
        if completed_games > SELF_PLAY_GAMES - SIMUL_GAMES {
            should_play.store(false, Ordering::Relaxed);
        }
    }

    total_examples
}

/// Run a single game against self.
fn self_play_game<const N: usize, A: Agent<N>>(agent: &A) -> Vec<Example<N>>
where
    [[Option<Tile>; N]; N]: Default,
    Turn<N>: Lut,
{
    let mut game_examples = Vec::new();
    let mut game = Game::default(); // TODO add komi?

    // make random moves for the first few turns to diversify training data
    for _ in 0..OPENING_PLIES {
        game.nth_place(random()).unwrap();
    }

    // initialize MCTS
    let mut node = Node::default();
    while matches!(game.winner(), GameResult::Ongoing) {
        for _ in 0..ROLLOUTS_PER_MOVE {
            node.rollout(game.clone(), agent);
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
    println!(
        "{winner:?} in {} plies\n{}",
        game.ply, game.board
    );
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
