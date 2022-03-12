use alpha_tak::{
    agent::Agent,
    analysis::Analysis,
    config::{KOMI, N, ROLLOUTS_PER_MOVE, SELF_PLAY_GAMES, TEMPERATURE_PLIES},
    example::Example,
    model::network::Network,
    player::Player,
    threadpool::thread_pool,
};
use tak::*;

pub fn self_play(network: &Network<N>) -> Vec<Example<N>> {
    const WORKERS: usize = 128;

    let outputs = thread_pool::<N, WORKERS, _, _>(network, SELF_PLAY_GAMES, self_play_game);
    let mut examples = Vec::new();
    let mut analyses = Vec::new();
    for output in outputs {
        examples.extend(output.0.into_iter());
        analyses.push(output.1);
    }

    // TODO save analyses

    examples
}

fn self_play_game<A: Agent<N>>(agent: &A, _index: usize) -> (Vec<Example<N>>, Analysis<N>) {
    let mut game = Game::with_komi(KOMI);
    // TODO proper opening book using index
    let opening = game.opening(rand::random()).unwrap();

    let mut player = Player::new(agent, opening);

    while matches!(game.winner(), GameResult::Ongoing) {
        player.rollout(&game, ROLLOUTS_PER_MOVE);
        let turn = player.pick_move(&game, game.ply > TEMPERATURE_PLIES);
        player.play_move(&game, &turn);
        game.play(turn).unwrap();
    }

    (player.get_examples(game.winner()), player.get_analysis())
}
