use std::{collections::VecDeque, sync::Arc, thread::spawn};

use crossbeam::channel::{bounded, Receiver, Sender};
use tak::*;

use crate::{
    analysis::Analysis,
    example::{Example, IncompleteExample},
    model::network::Network,
    search::{node::Node, turn_map::Lut},
};

pub struct BatchPlayer<const N: usize> {
    node: Node<N>,
    network: Arc<Network<N>>,
    batch_size: usize,
    pipeline_depth: usize,
    work_tx: Sender<(Vec<Game<N>>, Sender<(Vec<Vec<f32>>, Vec<f32>)>)>,
    batch_queue: VecDeque<(Vec<Vec<Turn<N>>>, Receiver<(Vec<Vec<f32>>, Vec<f32>)>)>,
    examples: Vec<IncompleteExample<N>>,
    analysis: Analysis<N>,
}

impl<const N: usize> BatchPlayer<N>
where
    Turn<N>: Lut,
{
    fn send_work(&mut self, game: &Game<N>) {
        let (paths, games): (Vec<_>, Vec<_>) = (0..self.batch_size)
            .filter_map(|_| {
                let mut path = vec![];
                let mut game = game.clone();
                if self.node.virtual_rollout(&mut game, &mut path) == GameResult::Ongoing {
                    Some((path, game))
                } else {
                    None
                }
            })
            .unzip();

        let (tx, rx) = bounded(1);
        self.work_tx.send((games, tx)).unwrap();
        self.batch_queue.push_back((paths, rx));
    }

    fn fill_pipeline(&mut self, game: &Game<N>) {
        for _ in 0..self.pipeline_depth - self.batch_queue.len() {
            self.send_work(game);
        }
    }

    fn process_batch(&mut self) {
        let (paths, batch_rx) = self.batch_queue.pop_front().unwrap();
        let (policy_vecs, evals) = batch_rx.recv().unwrap();

        policy_vecs
            .into_iter()
            .zip(evals)
            .zip(paths)
            .for_each(|(result, path)| {
                self.node.devirtualize_path(&mut path.into_iter(), &result);
            });
    }

    fn process_pipeline(&mut self) {
        while !self.batch_queue.is_empty() {
            self.process_batch();
        }
    }

    pub fn new(
        game: &Game<N>,
        network: Network<N>,
        opening: Vec<Turn<N>>,
        komi: i32,
        batch_size: usize,
        pipeline_depth: usize,
    ) -> Self {
        let network = Arc::new(network);
        let (work_tx, work_rx) = bounded::<(Vec<_>, Sender<_>)>(pipeline_depth);

        for _ in 0..pipeline_depth {
            let network = network.clone();
            let work_rx = work_rx.clone();
            spawn(move || {
                while let Ok((games, batch_tx)) = work_rx.recv() {
                    batch_tx
                        .send(if games.is_empty() {
                            Default::default()
                        } else {
                            network.policy_eval_batch(games.as_slice())
                        })
                        .unwrap();
                }
            });
        }

        let mut instance = Self {
            node: Default::default(),
            network,
            batch_size,
            pipeline_depth,
            work_tx,
            batch_queue: Default::default(),
            examples: Vec::new(),
            analysis: Analysis::from_opening(opening, komi),
        };

        instance.fill_pipeline(game);

        instance
    }

    pub fn debug(&self, limit: Option<usize>) -> String {
        self.node.debug(limit)
    }

    /// Do a batch of rollouts.
    pub fn rollout(&mut self, game: &Game<N>) {
        self.send_work(game);
        self.process_batch();
    }

    /// Pick a move to play and also play it.
    pub fn pick_move(&mut self, game: &Game<N>, exploitation: bool) -> Turn<N> {
        let turn = self.node.pick_move(exploitation);
        self.play_move(game, &turn);
        turn
    }

    /// Update the search tree, analysis, and create an example.
    pub fn play_move(&mut self, game: &Game<N>, turn: &Turn<N>) {
        // rollout stale paths
        // necessary to update policies accordingly
        // TODO: avoid rolling out nodes that are going to be discarded
        self.process_pipeline();

        // save example
        self.examples.push(IncompleteExample {
            game: game.clone(),
            policy: self.node.improved_policy(),
        });

        self.analysis.update(&self.node, turn.clone());

        self.node = std::mem::take(&mut self.node).play(turn);

        // refill queue
        let mut game = game.clone();
        game.play(turn.clone()).unwrap();
        self.fill_pipeline(&game);
    }

    /// Complete collected examples with the game result and return them.
    /// The examples in the Player will be empty after this method is used.
    pub fn get_examples(&mut self, result: GameResult) -> Vec<Example<N>> {
        let white_result = match result {
            GameResult::Winner {
                colour: Colour::White,
                ..
            } => 1.,
            GameResult::Winner {
                colour: Colour::Black,
                ..
            } => -1.,
            GameResult::Draw { .. } => 0.,
            GameResult::Ongoing { .. } => unreachable!(
                "cannot complete examples
    with ongoing game"
            ),
        };
        std::mem::take(&mut self.examples)
            .into_iter()
            .map(|ex| {
                let perspective = if ex.game.to_move == Colour::White {
                    white_result
                } else {
                    -white_result
                };
                ex.complete(perspective)
            })
            .collect()
    }

    /// Get the analysis of the game
    pub fn get_analysis(&mut self) -> Analysis<N> {
        std::mem::take(&mut self.analysis)
    }

    /// Apply dirichlet noise to the top node
    pub fn apply_dirichlet(&mut self, game: &Game<N>, alpha: f32, ratio: f32) {
        self.node.rollout(game.clone(), self.network.as_ref());
        self.node.apply_dirichlet(alpha, ratio);
    }
}
