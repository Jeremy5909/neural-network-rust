use matrix::Matrix;
use network::SIGMOID;

use crate::network::Network;

mod matrix;
mod macros;
mod network;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    
    let mut network = Network::new(vec![2,3,1], SIGMOID, 0.5);

    network.train(inputs, targets, 10000);

    println!("0 XOR 0: {}", network.feed_forward(Matrix::from(vec![0.0, 0.0])));
}
