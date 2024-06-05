mod infer;
mod train;

use autodiff::*;
use mnist::MnistBuilder;

fn argmax(x: &[f32]) -> usize {
    let mut max = 0.;
    let mut max_idx = 0;
    for (i, &x) in x.iter().enumerate() {
        if x > max {
            max = x;
            max_idx = i;
        }
    }
    max_idx
}

fn main() {
    let s = 28 * 28;
    let e_size = 3;

    let mut nn = Value::Matrix(DMatrix::<f32>::new_random(s, 1)).symbol("x");
    let target = nn.push(Node::Parameter(
        autodiff::Value::Matrix(DMatrix::<f32>::new_random(10, 1)),
        "target".to_string(),
    ));

    // Encoder

    let encoder_w = Value::Matrix(
        DMatrix::<f32>::new_random(e_size, s + 10).map(|n| n - 0.5) / (28. * 28. / 2.),
    )
    .symbol("encoder_w");

    //let encoder_b =
    //    Value::Matrix(DMatrix::<f32>::new_random(3, 1).map(|n| n - 0.5) / 2.).symbol("encoder_b");

    nn.push(Node::AppendVert(
        nn.get_symbol("x"),
        nn.get_symbol("target"),
    ));
    nn = encoder_w.matmul(nn).symbol("encoder_output").tanh();

    // Decoder

    let decoder_w = Value::Matrix(
        DMatrix::<f32>::new_random(s, e_size + 10).map(|n| n - 0.5) / ((28. * 28. + 10.) / 2.),
    )
    .symbol("decoder_w");

    //let decoder_b =
    //    Value::Matrix(DMatrix::<f32>::new_random(s, 1).map(|n| n - 0.5) / 2.).symbol("decoder_b");

    nn.push(Node::AppendVert(
        nn.get_symbol("encoder_output"),
        nn.get_symbol("target"),
    ));
    nn = decoder_w.matmul(nn).symbol("decoder_output").tanh();

    // Backprop

    let decoder_dy = nn.push(Node::Sub(
        nn.get_symbol("decoder_output"),
        nn.get_symbol("x"),
    ));

    for (s, k) in nn.symbols.iter() {
        println!("{s} = {:?}", k)
    }

    if let Ok(model) = std::env::var("TRAIN") {
        train::train(&mut nn, model, &decoder_dy);
        return;
    }

    print!("Loading model.bin... ");
    let mut nn = Graph::from_serialisable(
        bincode::deserialize_from(std::fs::File::open("model.bin").unwrap()).unwrap(),
    );
    println!("ok");

    infer::infer(&mut nn);
}
// - without caching : 1min + 10s
// - with caching    : 40s
