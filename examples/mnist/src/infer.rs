use autodiff::*;
use minifb::{Key::*, Window, WindowOptions};

pub fn infer(nn: &mut Graph) {
    let s = 28 * 28;

    let mut window = Window::new(
        "Test - ESC to exit",
        28,
        28,
        WindowOptions {
            resize: true,
            scale: minifb::Scale::X8,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let nums = [Key0, Key1, Key2, Key3, Key4, Key5, Key6, Key7, Key8, Key9];
    let mut style_dim = 0;

    *nn.parameter_mut("target") = Value::Matrix(DMatrix::<f32>::from_iterator(
        10,
        1,
        (0..10).map(|i| if i == 0 { 1. } else { 0. }),
    ));

    let eo = nn.get_symbol("encoder_output");
    nn.nodes[eo.0] = Node::Parameter(
        Value::Matrix(DMatrix::<f32>::new_random(3, 1).map(|n| n * 2. - 1.)),
        "encoder_output".to_string(),
    );

    loop {
        if window.is_key_down(Q) {
            break;
        }
        for k in nums {
            if window.is_key_pressed(k, minifb::KeyRepeat::No) {
                *nn.parameter_mut("target") = Value::Matrix(DMatrix::<f32>::from_iterator(
                    10,
                    1,
                    (0..10).map(|i| if i == k as usize { 1. } else { 0. }),
                ));
            }
        }

        let eo = nn.get_symbol("encoder_output");
        if let Node::Parameter(m, _) = &mut nn.nodes[eo.0] {
            if window.is_key_pressed(D, minifb::KeyRepeat::No) {
                style_dim = (style_dim + 1) % m.mat().nrows();
            }
            if window.is_key_pressed(J, minifb::KeyRepeat::Yes) {
                m.mat_mut()[style_dim] += 0.1;
            }
            if window.is_key_pressed(K, minifb::KeyRepeat::Yes) {
                m.mat_mut()[style_dim] -= 0.1;
            }
        }

        let data = nn.eval(&nn.get_symbol("decoder_output")).mat().as_slice();

        let max = data.iter().fold(0., |a: f32, &b| a.max(b.abs()));

        window
            .update_with_buffer(
                &data
                    .iter()
                    .map(|n| {
                        if n < &0. {
                            ((n.abs() / max * 254.) as u32) << 16
                        } else {
                            (n / max * 254.) as u32
                        }
                    })
                    .collect::<Vec<_>>(),
                28,
                28,
            )
            .unwrap();

        nn.clear_cache();
    }
}
