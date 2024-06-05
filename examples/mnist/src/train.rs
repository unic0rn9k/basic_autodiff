use crate::argmax;
use autodiff::*;
use mnist::MnistBuilder;

pub fn train(nn: &mut Graph, model: String, decoder_dy: &NodeId) {
    let ntrain: usize = 60000;
    let epochs = 8;
    let ntest: usize = 200;
    let s = 28 * 28;

    let mnist::Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data/")
        //.training_images_filename("emnist-mnist-train-images-idx3-ubyte")
        //.training_labels_filename("emnist-mnist-train-labels-idx1-ubyte")
        .training_set_length(ntrain as u32)
        .test_set_length(ntest as u32)
        .download_and_extract()
        .finalize();

    let [ddw, dew] = nn.derive(
        &nn.get_symbol("decoder_output"),
        ["decoder_w", "encoder_w"],
        decoder_dy,
    );

    let mut lr = 0.0002;

    for n in 0..ntrain * epochs {
        let dp = n % ntrain;

        *nn.parameter_mut("x") = Value::Matrix(DMatrix::<f32>::from_iterator(
            s,
            1,
            trn_img[dp * s..dp * s + s].iter().map(|x| *x as f32 / 255.),
        ));

        *nn.parameter_mut("target") = Value::Matrix(DMatrix::<f32>::from_iterator(
            10,
            1,
            (0..10).map(|i| if i == trn_lbl[dp] as usize { 1. } else { 0. }),
        ));

        //let dw = nn.eval(dw).clone();
        //let db = nn.eval(db).clone();
        let ddw = nn.eval(&ddw).clone();
        let dew = nn.eval(&dew).clone();

        if ddw.mat().iter().chain(dew.mat().iter()).any(|x| x.is_nan())
            || dew.mat().iter().any(|x| x.is_nan())
        {
            println!("NaN detected");
            break;
        }

        //*nn.parameter_mut("w").mat_mut() -= dw.mat().clone() * lr;
        //*nn.parameter_mut("b").mat_mut() -= db.mat().clone() * lr;
        *nn.parameter_mut("decoder_w").mat_mut() -= ddw.mat().clone() * lr;
        *nn.parameter_mut("encoder_w").mat_mut() -= dew.mat().clone() * lr;

        if n % 1000 == 0 {
            println!("{:.2}%", n as f32 / ntrain as f32 / epochs as f32 * 100.);
        }

        lr *= 0.99999;
        nn.clear_cache()
    }

    bincode::serialize_into(std::fs::File::create(model).unwrap(), &nn.serializable()).unwrap();
}
