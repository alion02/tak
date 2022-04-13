use std::{collections::HashMap, thread::spawn, time::Instant};

use rand_distr::{Distribution, WeightedIndex};
use tak::*;

use super::node::Node;

impl<const N: usize> Node<N> {
    fn check_initialized(&self) {
        assert!(self.is_initialized(), "node must be initialized");
    }

    pub fn improved_policy(&self) -> HashMap<Turn<N>, u32> {
        self.check_initialized();
        // after many rollouts the visit counts become a better estimate
        // for policy (not normalized)
        HashMap::from_iter(
            self.children
                .iter()
                .map(|(turn, child)| (turn.clone(), child.visits)),
        )
    }

    #[must_use]
    pub fn play(mut self, turn: &Turn<N>) -> Node<N> {
        self.check_initialized();
        let child = self
            .children
            .remove(turn)
            .expect("attempted to play invalid move");

        spawn(move || {
            fn verify_invariants<const N: usize>(node: &Node<N>) {
                assert_eq!(node.virtual_visits, 0);
                match node.visits {
                    0 => {
                        assert_eq!(node.result, GameResult::Ongoing);
                        assert_eq!(node.children.len(), 0);
                    }
                    _ => {
                        assert!((node.result == GameResult::Ongoing) ^ (node.children.len() == 0));
                    }
                }
                node.children.values().for_each(verify_invariants);
            }

            fn count_nodes<const N: usize>(node: &Node<N>) -> u64 {
                node.children.values().map(count_nodes).sum::<u64>() + 1
            }

            fn count_single_visit<const N: usize>(node: &Node<N>) -> u64 {
                if node.visits == 1 {
                    1
                } else {
                    node.children.values().map(count_single_visit).sum()
                }
            }

            verify_invariants(&self);

            let to_drop_nodes = count_nodes(&self);
            let single_visit_nodes = count_single_visit(&self);

            println!(
                "{} nodes to be dropped\n{} rollouts\n{} single visit",
                to_drop_nodes, self.visits, single_visit_nodes
            );

            let start = Instant::now();

            drop(self);

            let time = start.elapsed();
            println!(
                "Deallocated nodes: {} in {}ms ({}/ms)",
                to_drop_nodes,
                time.as_millis(),
                to_drop_nodes as f64 / (time.as_secs_f64() * 1000.0)
            );
        });

        child
    }

    pub fn pick_move(&self, exploitation: bool) -> Turn<N> {
        let improved_policy = self.improved_policy();

        if exploitation {
            // when exploiting always pick the move with largest policy
            improved_policy
                .into_iter()
                .max_by_key(|(_, value)| *value)
                .unwrap()
                .0
        } else {
            // split into turns and weights
            let mut turns = vec![];
            let mut weights = vec![];
            for (turn, weight) in improved_policy {
                turns.push(turn);
                weights.push(weight);
            }
            // randomly pick based on weights from improved policy
            let mut rng = rand::thread_rng();
            let distr = WeightedIndex::new(&weights).unwrap();
            let index = distr.sample(&mut rng);
            turns.swap_remove(index)
        }
    }
}
