use std::{collections::HashMap, fmt::Debug, unimplemented};

// ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
pub use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);

#[derive(Clone)]
pub enum Value {
    Matrix(DMatrix<f32>),
    Scalar(f32),
    None,
}

impl Value {
    pub fn none() -> Self {
        //println!("None!");
        Value::None
    }
}

#[derive(Debug)]
struct Dim(Option<(usize, usize)>);
impl PartialEq for Dim {
    fn eq(&self, other: &Self) -> bool {
        match (self.0, other.0) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Matrix(m) => write!(f, "[{}x{}]", m.nrows(), m.ncols()),
            Value::Scalar(s) => write!(f, "{s}"),
            Value::None => write!(f, "None"),
        }
    }
}

impl Value {
    pub fn argmax(&self) -> usize {
        if let Value::Matrix(m) = self {
            let mut max = 0;
            for (i, n) in m.iter().enumerate() {
                if n > &m[i] {
                    max = i
                }
            }
            return max;
        }
        panic!("Oh no")
    }

    pub fn ascii_mat(&self, r: usize, c: usize) -> String {
        let chars = "      .,:;%&#@";
        if let Value::Matrix(m) = self {
            let max = m.max();
            println!("max: {max}");
            let m = m
                .clone()
                .reshape_generic(nalgebra::Dyn(r), nalgebra::Dyn(c));
            let mut s = String::new();
            for i in 0..m.ncols() {
                for j in 0..m.nrows() {
                    s.push(
                        chars
                            .chars()
                            .nth(((m[(j, i)].max(0.) / max) * (chars.len() - 1) as f32) as usize)
                            .unwrap(),
                    );
                }
                s.push('\n');
            }
            s
        } else {
            panic!()
        }
    }

    fn dim(&self) -> Dim {
        match self {
            Value::Matrix(m) => Dim(Some((m.nrows(), m.ncols()))),
            _ => Dim(None),
        }
    }

    pub fn add(&self, other: &Value) -> Value {
        assert_eq!(self.dim(), other.dim());
        match (self, other) {
            (Value::Matrix(a), Value::Matrix(b)) => Value::Matrix(a + b),
            (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a + b),
            (Value::Matrix(a), Value::Scalar(b)) => Value::Matrix(a.map(|n| n + b)),
            (Value::Scalar(a), Value::Matrix(b)) => Value::Matrix(b.map(|n| n + a)),
            (Value::None, b) => b.clone(),
            (a, Value::None) => a.clone(),
        }
    }

    pub fn sub(&self, other: &Value) -> Value {
        assert_eq!(self.dim(), other.dim());
        match (self, other) {
            (Value::Matrix(a), Value::Matrix(b)) => Value::Matrix(a - b),
            (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a - b),
            (Value::Matrix(a), Value::Scalar(b)) => Value::Matrix(a.map(|n| n - b)),
            (Value::Scalar(a), Value::Matrix(b)) => Value::Matrix(b.map(|n| a - n)),
            (Value::None, b) => -b.clone(),
            (a, Value::None) => a.clone(),
            _ => unimplemented!(),
        }
    }

    pub fn mul(&self, other: &Value) -> Value {
        assert_eq!(self.dim(), other.dim());
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a * b),
            (Value::Matrix(a), Value::Scalar(b)) => Value::Matrix(a.map(|n| n * b)),
            (Value::Scalar(a), Value::Matrix(b)) => Value::Matrix(b.map(|n| a * n)),
            (Value::Matrix(a), Value::Matrix(b)) => Value::Matrix(a.component_mul(b)),
            _ => unimplemented!(),
        }
    }

    pub fn div(&self, other: &Value) -> Value {
        assert_eq!(self.dim(), other.dim());
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a / b),
            (Value::Matrix(a), Value::Scalar(b)) => Value::Matrix(a.map(|n| n / b)),
            (Value::Scalar(a), Value::Matrix(b)) => Value::Matrix(b.map(|n| a / n)),
            (Value::Matrix(a), Value::Matrix(b)) => Value::Matrix(a.component_div(b)),
            _ => unimplemented!(),
        }
    }

    pub fn matmul(&self, other: &Value) -> Value {
        match (self, other) {
            (Value::Matrix(a), Value::Matrix(b)) => {
                assert_eq!(a.ncols(), b.nrows(), "{:?} !* {:?}", a.shape(), b.shape());
                Value::Matrix(a * b)
            },
            (Value::None, _) | (_, Value::None) => Value::Scalar(0.),
            _ => unimplemented!()
            //(Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a * b),
            //(Value::Matrix(a), Value::Scalar(b)) => Value::Matrix(a.map(|n| n * b)),
            //(Value::Scalar(a), Value::Matrix(b)) => Value::Matrix(b.map(|n| a * n)),
        }
    }

    pub fn sum(&self) -> Value {
        match self {
            Value::Matrix(a) => Value::Scalar(a.sum()),
            Value::Scalar(a) => Value::Scalar(*a),
            _ => unimplemented!(),
        }
    }

    pub fn exp(&self) -> Value {
        match self {
            Value::Matrix(a) => Value::Matrix(a.map(|x| x.exp())),
            Value::Scalar(a) => Value::Scalar(a.exp()),
            _ => unimplemented!(),
        }
    }

    pub fn transpose(&self) -> Value {
        match self {
            Value::Matrix(a) => Value::Matrix(a.transpose()),
            _ => unimplemented!(),
        }
    }

    pub fn tanh(&self) -> Value {
        match self {
            Value::Matrix(a) => Value::Matrix(a.map(|x| x.tanh())),
            Value::Scalar(a) => Value::Scalar(a.tanh()),
            _ => unimplemented!(),
        }
    }

    pub fn symbol(self, symbol: &'static str) -> Graph {
        Graph {
            symbols: [(symbol.to_string(), NodeId(0))].into_iter().collect(),
            nodes: vec![Node::Parameter(self, symbol.to_string())],
            leaf: NodeId(0),
            value_cache: HashMap::new(),
            node_cache: HashMap::new(),
        }
    }

    pub fn mat(&self) -> &DMatrix<f32> {
        match self {
            Value::Matrix(a) => a,
            n => panic!("Called `mat` on {n:?}"),
        }
    }

    pub fn mat_mut(&mut self) -> &mut DMatrix<f32> {
        match self {
            Value::Matrix(a) => a,
            n => panic!("Called `mat_mut` on {n:?}"),
        }
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        match self {
            Value::Matrix(m) => Value::Matrix(-1. * m),
            Value::Scalar(s) => Value::Scalar(-s),
            Value::None => Value::none(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Node {
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    MatMul(NodeId, NodeId),
    Sum(NodeId),
    Exp(NodeId),
    Transpose(NodeId),
    Parameter(Value, String),
    Neg(NodeId),
    AppendVert(NodeId, NodeId),
    SubSlice(NodeId, usize, usize, usize, usize),
    Tanh(NodeId),
}

impl Node {
    fn hash(&self) -> (usize, usize, usize) {
        match self {
            Node::Add(a, b) => (0, a.0, b.0),
            Node::Sub(a, b) => (1, a.0, b.0),
            Node::Mul(a, b) => (2, a.0, b.0),
            Node::Div(a, b) => (3, a.0, b.0),
            Node::MatMul(a, b) => (4, a.0, b.0),
            Node::Sum(a) => (5, a.0, 0),
            Node::Exp(a) => (6, a.0, 0),
            Node::Transpose(a) => (7, a.0, 0),
            Node::Neg(a) => (8, a.0, 0),
            Node::Parameter(_, _) => (9, 0, 0),
            Node::AppendVert(a, b) => (10, a.0, b.0),
            Node::SubSlice(a, b, c, d, e) => (11, a.0, b * 1000000 + c * 10000 + d * 100 + e),
            Node::Tanh(a) => (12, a.0, 0),
        }
    }

    fn wrt(&self) -> Option<&str> {
        if let Node::Parameter(_, wrt) = self {
            Some(wrt)
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct Graph {
    pub symbols: HashMap<String, NodeId>,
    value_cache: HashMap<usize, Value>,
    node_cache: HashMap<(usize, usize, usize), NodeId>,
    pub nodes: Vec<Node>,
    leaf: NodeId,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            symbols: HashMap::new(),
            value_cache: HashMap::new(),
            node_cache: HashMap::new(),
            nodes: Vec::new(),
            leaf: NodeId(0),
        }
    }

    pub fn push(&mut self, node: Node) -> NodeId {
        self.leaf = NodeId(self.nodes.len());
        if self.node_cache.contains_key(&node.hash()) {
            self.node_cache[&node.hash()]
        } else {
            let id = NodeId(self.nodes.len());
            let hash = node.hash();
            self.nodes.push(node);
            if let Node::Parameter(_, s) = &self.nodes[id.0] {
                self.symbols.insert(s.to_string(), id);
            } else {
                self.node_cache.insert(hash, id);
            }
            id
        }
    }

    pub fn eval(&mut self, id: &NodeId) -> &Value {
        //println!("{}\t: {:?}", id.0, self.nodes[id.0]);
        use Node::*;
        if self.value_cache.contains_key(&id.0) {
            return &self.value_cache[&id.0];
        }
        let val = match self.nodes[id.0].clone() {
            Add(a, b) => self.eval(&a).clone().add(self.eval(&b)),
            Sub(a, b) => self.eval(&a).clone().sub(self.eval(&b)),
            Mul(a, b) => self.eval(&a).clone().mul(self.eval(&b)),
            Div(a, b) => self.eval(&a).clone().div(self.eval(&b)),
            MatMul(a, b) => self.eval(&a).clone().matmul(self.eval(&b)),
            Sum(a) => self.eval(&a).sum(),
            Exp(a) => self.eval(&a).exp(),
            Transpose(v) => self.eval(&v).transpose(),
            Parameter(a, _) => a,
            Neg(a) => self.eval(&a).mul(&Value::Scalar(-1.0)),
            AppendVert(a, b) => {
                let a = self.eval(&a).mat().clone();
                let b = self.eval(&b).mat();
                assert_eq!(a.ncols(), b.ncols());
                Value::Matrix(DMatrix::from_iterator(
                    a.nrows() + b.nrows(),
                    a.ncols(),
                    a.iter().chain(b.iter()).copied(),
                ))
            }
            SubSlice(a, b, c, d, e) => {
                let a = self.eval(&a).mat().clone();
                Value::Matrix(a.view((b, c), (d, e)).map(|x| x))
            }
            Tanh(a) => self.eval(&a).tanh(),
        };
        self.value_cache.insert(id.0, val);
        &self.value_cache[&id.0]
    }

    pub fn clear_cache(&mut self) {
        self.value_cache.clear()
    }

    pub fn derive<const LEN: usize>(
        &mut self,
        at: &NodeId,
        wrt: [&'static str; LEN],
        grad: &NodeId,
    ) -> [NodeId; LEN] {
        use Node::*;
        let mut res = [NodeId(0); LEN];

        for (i, wrt) in wrt.into_iter().enumerate() {
            res[i] = match self.nodes[at.0] {
                Mul(a, b) => {
                    let a_grad = self.push(Mul(b, *grad));
                    let b_grad = self.push(Mul(a, *grad));
                    let n = Add(
                        self.derive(&a, [wrt], &a_grad)[0],
                        self.derive(&b, [wrt], &b_grad)[0],
                    );
                    self.push(n)
                }

                Add(a, b) => {
                    let n = Add(
                        self.derive(&a, [wrt], grad)[0],
                        self.derive(&b, [wrt], grad)[0],
                    );
                    self.push(n)
                }

                Sub(a, b) => {
                    let n = Sub(
                        self.derive(&a, [wrt], grad)[0],
                        self.derive(&b, [wrt], grad)[0],
                    );
                    self.push(n)
                }

                Div(a, b) => {
                    let a_grad = self.push(Div(*grad, b));
                    let n = Div(a, self.push(Mul(b, b)));
                    let n = self.push(n);
                    let b_grad = self.push(Mul(n, *grad));
                    let n = Sub(
                        self.derive(&a, [wrt], &a_grad)[0],
                        self.derive(&b, [wrt], &b_grad)[0],
                    );
                    self.push(n)
                }

                MatMul(a, b) => {
                    let n = self.push(Transpose(b));
                    let a_grad = self.push(MatMul(*grad, n));
                    let n = self.push(Transpose(a));
                    let b_grad = self.push(MatMul(n, *grad));
                    let n = Add(
                        self.derive(&a, [wrt], &a_grad)[0],
                        self.derive(&b, [wrt], &b_grad)[0],
                    );
                    self.push(n)
                }

                Sum(a) => self.derive(&a, [wrt], grad)[0],
                Exp(a) => {
                    let n = Mul(self.push(Exp(a)), *grad);
                    let n = self.push(n);
                    self.derive(&a, [wrt], &n)[0]
                }
                Transpose(_) => unimplemented!(),
                Neg(a) => {
                    let n = self.push(Neg(*grad));
                    self.derive(&a, [wrt], &n)[0]
                }

                Parameter(_, _) => {
                    if self.nodes[at.0].wrt() == Some(wrt) {
                        *grad
                    } else {
                        self.push(Node::Parameter(Value::none(), "".to_string()))
                    }
                }

                AppendVert(a, b) => {
                    let a_shape = self.eval(&a).mat().shape();
                    let b_shape = self.eval(&b).mat().shape();
                    let a_grad = self.push(SubSlice(*grad, 0, 0, a_shape.0, a_shape.1));
                    let b_grad =
                        self.push(SubSlice(*grad, a_shape.0, a_shape.1, b_shape.0, b_shape.1));
                    let n = Add(
                        self.derive(&a, [wrt], &a_grad)[0],
                        self.derive(&b, [wrt], &b_grad)[0],
                    );
                    self.push(n)
                }

                SubSlice(_, _, _, _, _) => unimplemented!(),

                Tanh(a) => {
                    let dt = self.push(Mul(*at, *at));
                    let dt = self.push(Neg(dt));
                    let one = self.push(Node::Parameter(Value::Scalar(1.0), "1".to_string()));
                    let dt = self.push(Add(one, dt));
                    let n = Mul(dt, *grad);
                    let n = self.push(n);
                    self.derive(&a, [wrt], &n)[0]
                }
            }
        }

        res
    }

    pub fn value(&mut self) -> &Value {
        self.eval(&self.leaf.clone())
    }

    pub fn leaf(&self) -> NodeId {
        self.leaf
    }

    pub fn exp(mut self) -> Graph {
        use Node::*;
        let id = self.push(Exp(self.leaf));
        Graph {
            symbols: self.symbols,
            value_cache: self.value_cache,
            node_cache: self.node_cache,
            nodes: self.nodes,
            leaf: id,
        }
    }

    pub fn sum(mut self) -> Graph {
        use Node::*;
        let id = self.push(Sum(self.leaf));
        Graph {
            symbols: self.symbols,
            value_cache: self.value_cache,
            node_cache: self.node_cache,
            nodes: self.nodes,
            leaf: id,
        }
    }

    fn assimilate(&mut self, sym: HashMap<String, NodeId>) {
        let offset = self.nodes.len();
        self.symbols
            .extend(sym.into_iter().map(|(s, id)| (s, NodeId(id.0 + offset))))
    }

    pub fn parameter(&self, sym: &'static str) -> &Value {
        match &self.nodes[self.symbols[sym].0] {
            Node::Parameter(p, _) => p,
            _ => panic!("Symbol has to be a parameter"),
        }
    }
    pub fn parameter_mut(&mut self, sym: &'static str) -> &mut Value {
        //self.clear_cache();
        match &mut self.nodes[self.symbols[sym].0] {
            Node::Parameter(p, _) => p,
            _ => panic!("Symbol has to be a parameter"),
        }
    }
    pub fn new_symbol(&mut self, sym: &'static str) -> NodeId {
        self.push(Node::Parameter(Value::none(), sym.to_string()))
    }

    pub fn print(&self) {
        for (i, n) in self.nodes.iter().enumerate() {
            println!("{}: {:?}", i, n);
        }
    }

    pub fn tanh(mut self) -> Graph {
        use Node::*;
        let id = self.push(Tanh(self.leaf));
        Graph {
            symbols: self.symbols,
            value_cache: self.value_cache,
            node_cache: self.node_cache,
            nodes: self.nodes,
            leaf: id,
        }
    }

    pub fn symbol(mut self, sym: &'static str) -> Self {
        self.symbols.insert(sym.to_string(), self.leaf);
        self
    }

    pub fn get_symbol(&self, sym: &'static str) -> NodeId {
        self.symbols[sym]
    }

    pub fn serializable(&self) -> SerializableGraph {
        let mut parameters = vec![];
        let nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|n| {
                if let Node::Parameter(p, _) = n {
                    parameters.push(match p {
                        Value::Matrix(m) => {
                            SerializableValue::Matrix(m.data.as_vec().clone(), m.nrows(), m.ncols())
                        }
                        Value::Scalar(n) => SerializableValue::Scalar(*n),
                        Value::None => SerializableValue::None,
                    });
                }
                n.hash()
            })
            .collect();

        (
            nodes,
            self.symbols.clone(),
            parameters,
            self.node_cache.clone(),
            self.leaf,
        )
    }

    //Node::Add(a, b) => (0, a.0, b.0),
    //Node::Sub(a, b) => (1, a.0, b.0),
    //Node::Mul(a, b) => (2, a.0, b.0),
    //Node::Div(a, b) => (3, a.0, b.0),
    //Node::MatMul(a, b) => (4, a.0, b.0),
    //Node::Sum(a) => (5, a.0, 0),
    //Node::Exp(a) => (6, a.0, 0),
    //Node::Transpose(a) => (7, a.0, 0),
    //Node::Neg(a) => (8, a.0, 0),
    //Node::Parameter(_, _) => (9, 0, 0),
    //Node::AppendVert(a, b) => (10, a.0, b.0),
    //Node::SubSlice(a, b, c, d, e) => (11, a.0, b * 1000000 + c * 10000 + d * 100 + e),
    //Node::Tanh(a) => (12, a.0, 0),
    pub fn from_serialisable(s: SerializableGraph) -> Self {
        let mut graph = Graph::new();
        let (nodes, symbols, mut parameters, node_cache, leaf) = s;
        for n in nodes.into_iter() {
            let node = match n {
                (0, a, b) => Node::Add(NodeId(a), NodeId(b)),
                (1, a, b) => Node::Sub(NodeId(a), NodeId(b)),
                (2, a, b) => Node::Mul(NodeId(a), NodeId(b)),
                (3, a, b) => Node::Div(NodeId(a), NodeId(b)),
                (4, a, b) => Node::MatMul(NodeId(a), NodeId(b)),
                (5, a, _) => Node::Sum(NodeId(a)),
                (6, a, _) => Node::Exp(NodeId(a)),
                (7, a, _) => Node::Transpose(NodeId(a)),
                (8, a, _) => Node::Neg(NodeId(a)),
                (9, _, _) => {
                    let p = match parameters.remove(0) {
                        SerializableValue::Scalar(n) => Value::Scalar(n),
                        SerializableValue::Matrix(m, r, c) => {
                            Value::Matrix(DMatrix::from_column_slice(r, c, &m[..]))
                        }
                        SerializableValue::None => Value::None,
                    };
                    Node::Parameter(p, "".to_string())
                }
                (10, a, b) => Node::AppendVert(NodeId(a), NodeId(b)),

                (11, a, b) => {
                    let e = b % 100;
                    let d = (b / 100) % 100;
                    let c = (b / 10000) % 100;
                    let b = b / 1000000;
                    Node::SubSlice(NodeId(a), b, c, d, e)
                }
                (12, a, _) => Node::Tanh(NodeId(a)),
                _ => panic!("Unknown node type"),
            };
            graph.nodes.push(node);
        }

        for (sym, id) in symbols.iter() {
            if let Node::Parameter(_, s) = &mut graph.nodes[id.0] {
                *s = sym.clone()
            }
        }

        graph.symbols = symbols;
        graph.leaf = leaf;
        graph.node_cache = node_cache;
        graph
    }
}

pub type SerializableGraph = (
    Vec<(usize, usize, usize)>,
    HashMap<String, NodeId>,
    Vec<SerializableValue>,
    HashMap<(usize, usize, usize), NodeId>,
    NodeId,
);

#[derive(Serialize, Deserialize)]
pub enum SerializableValue {
    Scalar(f32),
    Matrix(Vec<f32>, usize, usize),
    None,
}

pub fn softmax(mut graph: Graph) -> Graph {
    use Node::*;

    let exp = graph.push(Exp(graph.leaf));
    let sum = graph.push(Sum(exp));
    graph.push(Div(exp, sum));
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bruh_softmax() {
        use Node::*;

        let mut g = Value::Matrix(DMatrix::from_row_slice(2, 1, &[1.0, 2.0])).symbol("x");
        let x = g.leaf();

        let exp = g.push(Exp(x));
        let hun = g.push(Parameter(Value::Scalar(1000.0), "hun".to_string()));
        let exp_h = g.push(Mul(exp, hun));
        let sum = g.push(Sum(exp));
        let sm = g.push(Div(exp_h, sum));
        let one = g.push(Parameter(Value::Scalar(1.0), "one".to_string()));

        let delta = g.derive(&sm, ["x"], &one);
        println!("bruh: {:?}", g.eval(&delta[0]))
    }

    #[test]
    fn softmax() {
        use Node::*;

        let mut g = Value::Matrix(DMatrix::from_row_slice(2, 1, &[1.0, 2.0])).symbol("x");
        let one = g.push(Parameter(Value::Scalar(1.0), "one".to_string()));
        let mut sm = crate::softmax(g);

        let delta = sm.derive(&sm.leaf(), ["x"], &one);
        println!("sm: {:?}", sm.eval(&delta[0]))
    }
}

use std::ops::*;
pub trait MatMul {
    type Output;
    fn matmul(self, other: Self) -> Self::Output;
}

macro_rules! impl_op {
    ($($Op:ident:$op:ident),*) => {$(
        impl $Op for Graph {
            type Output = Graph;

            fn $op(mut self, other: Graph) -> Graph {
                use Node::*;
                let offset = self.nodes.len();

                self.assimilate(other.symbols);
                self.nodes
                    .extend(other.nodes.into_iter().map(|node| match node {
                        Add(a, b) => Add(NodeId(a.0 + offset), NodeId(b.0 + offset)),
                        Sub(a, b) => Sub(NodeId(a.0 + offset), NodeId(b.0 + offset)),
                        Mul(a, b) => Mul(NodeId(a.0 + offset), NodeId(b.0 + offset)),
                        Div(a, b) => Div(NodeId(a.0 + offset), NodeId(b.0 + offset)),
                        MatMul(a, b) => MatMul(NodeId(a.0 + offset), NodeId(b.0 + offset)),
                        Sum(a) => Sum(NodeId(a.0 + offset)),
                        Exp(a) => Exp(NodeId(a.0 + offset)),
                        Transpose(a) => Transpose(NodeId(a.0 + offset)),
                        Parameter(a,s) => Parameter(a,s),
                        Neg(a) => Neg(NodeId(a.0 + offset)),
                        AppendVert(a, b) => AppendVert(NodeId(a.0 + offset), NodeId(b.0 + offset)),
                        SubSlice(a, b, c, d, e) => SubSlice(
                            NodeId(a.0 + offset),b,c,d,e
                        ),
                        Tanh(a) => Tanh(NodeId(a.0 + offset)),
                    }));

                let b = NodeId(other.leaf.0 + offset);
                let a = self.leaf;
                self.nodes
                    .push($Op(a, b));

                self.leaf=NodeId(self.nodes.len()-1);
                self
            }
        }
    )*};
}

impl_op!(Add: add, Mul: mul, Sub: sub, Div: div, MatMul: matmul);
