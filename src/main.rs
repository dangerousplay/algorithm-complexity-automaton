use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::read_to_string;
use std::io;
use std::iter::Map;
use std::path::Path;
use std::rc::Rc;
use eyre::{bail, ensure, Report, Result, WrapErr};
use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;

const SIGMA: char = '&';

// Symbol represents a symbol from the alphabet
#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum Symbol {
    Character(char),
    // Sigma for non determinism
    Sigma
}

impl From<char> for Symbol {
    fn from(value: char) -> Self {
        match value {
            SIGMA => Symbol::Sigma,
            _ => Symbol::Character(value)
        }
    }
}

impl Symbol {
    pub fn parse<T: Into<Cow<'static, str>>>(text: T) -> eyre::Result<Self> {
        let text = text.into();
        
        ensure!(text.len() == 1, "Invalid symbol: {}", text);
        
        let symbol = text.chars().nth(0).expect("bug in symbol parse");

        Ok(Symbol::from(symbol))
    }
}

impl Into<char> for Symbol {
    fn into(self) -> char {
        match self {
            Symbol::Character(c) => c,
            Symbol::Sigma => SIGMA
        }
    }
}

impl ToString for Symbol {
    fn to_string(&self) -> String {
        match self { 
            Symbol::Character(c) => c.to_string(),
            Symbol::Sigma => SIGMA.to_string()
        }
    }
}

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Transition<'a> {
    pub from_state: Cow<'a, str>,
    pub to_state: Cow<'a, str>,
    pub symbol: Symbol
}

impl <'a> Transition<'a> {
    
    // q2 q1 a
    pub fn parse<T: Into<Cow<'a, str>>>(text: T) -> eyre::Result<Self> {
        let text = text.into();
        let transition = text.split_whitespace()
            .map(|f| f.to_string())
            .collect::<Vec<String>>();
        
        ensure!(transition.len() == 3, "Invalid transition");
        
        let symbol = Symbol::parse(transition[2].clone())?;
        
        Ok(
            Self {
                from_state: transition[0].clone().into(),
                to_state: transition[1].clone().into(),
                symbol,
            }
        )
    }
}


pub type Transitions<'a> = HashMap<Symbol, Vec<Cow<'a, str>>>;

#[derive(Clone, Debug)]
pub struct State<'a> {
    pub name: Cow<'a, str>,
    pub is_final: bool,
    pub transitions: Transitions<'a>,
}

impl<'a> State<'a> {
    pub fn new<T: Into<Cow<'a, str>>>(name: T, is_final: bool) -> Self {
        Self { name: name.into(), is_final, transitions: HashMap::new() }
    }
}


// pub struct Node<'a> {
//     pub state: State<'a>,
//     pub transitions: Map<Symbol, Vec<State<'a>>>
// }


pub struct Automaton<'a> {
    pub initial_state: State<'a>,
    pub alphabet: Vec<char>,
    pub graph: HashMap<String, State<'a>>
}

impl<'a> Automaton<'a> {
    pub fn new(initial_state: State<'a>, alphabet: Vec<char>, graph: HashMap<String, State<'a>>) -> Self {
        Self { initial_state, alphabet, graph }
    }
    
    pub fn from_file<P: AsRef<Path> + Display + Clone>(file: P) -> eyre::Result<Self> {
        let automaton_description = std::fs::read_to_string(file.clone())
            .wrap_err(format!("Failed to open automation file '{}'", file))?;
        
        return Automaton::from_text(automaton_description);
    }

    // - Linha 1 define o estado inicial
    // - Linha 2 define os símbolos do alfabeto
    // - Linha 3 define o conjunto de estados
    // - Linha 4 define o conjunto de estados finais
    // - Linhas 5-fim do arquivo definem as transições entre os estados
    pub fn from_text<D: Into<Cow<'a, str>>>(automaton_description: D) -> eyre::Result<Self> {
        let automaton_description = automaton_description.into();
        let lines = automaton_description
            .lines()
            .map(|f| f.trim_matches('\u{feff}').to_string())
            .collect::<Vec<String>>();

        ensure!(lines.len() >= 4, format!("The automaton file has less than 4 lines: {}", lines.len()));

        let initial_state_text = lines[0].to_string();
        let alphabet_text = lines[1].clone().split_whitespace().map(|f| f.to_string()).collect::<Vec<String>>();
        let states = lines[2].clone().split_whitespace().map(|f| f.to_string()).collect::<Vec<String>>();
        let final_states = lines[3].split_whitespace().map(|f| f.to_string()).collect::<Vec<String>>();
        let transitions_text = lines.iter().skip(4).clone().map(|f| f.to_string()).collect::<Vec<String>>();
        
        dbg!(initial_state_text.clone());
        
        ensure!(!initial_state_text.contains(' '), "Invalid initial state: '{}'", initial_state_text);
        ensure!(states.contains(&initial_state_text), "Initial state '{}' not declared in states", initial_state_text);
        
        for final_state in final_states.iter() {
            ensure!(states.contains(final_state), "Final state '{}' not declared in states", final_state);
        }
        
        let mut alphabet = Vec::new();

        for symbol in alphabet_text {
            let symbol: char = Symbol::parse(symbol)
                .wrap_err("Invalid character for alphabet")?
                .into();

            alphabet.push(symbol);
        }
        
        let states = states.into_iter()
            .map(|s| State::new(s.clone(), final_states.contains(&s)))
            .collect::<Vec<State<'static>>>();

        let mut graph: HashMap<String, State<'a>> = states.iter()
            .cloned()
            .map(|s| (s.name.to_string(), s))
            .collect();
        
        for transition_text in transitions_text {
            let transition = Transition::parse(transition_text)?;
            let symbol = transition.symbol.clone().into();
            let is_in_alphabet = alphabet.contains(&symbol);
            
            ensure!(is_in_alphabet, "Alphabet does not contain transition Symbol: {:?}", transition);
            ensure!(graph.contains_key(&transition.from_state.to_string()), "Invalid transition from state: {:?}", transition);
            ensure!(graph.contains_key(&transition.to_state.to_string()), "Invalid transition to state: {:?}", transition);
            
            let state = graph.get_mut(&transition.from_state.to_string())
                .expect("From state");
            let mut symbol_transitions = state.transitions
                .remove(&transition.symbol)
                .unwrap_or(Vec::new());
            
            symbol_transitions.push(transition.to_state);
            
            state.transitions.insert(transition.symbol, symbol_transitions);
        }

        let initial_state = graph.get(&initial_state_text).unwrap().clone();
        
        Ok(Self {
            initial_state,
            alphabet,
            graph: graph
        })
    }
}

impl<'a> From<&Automaton<'a>> for Graph {
    fn from(value: &Automaton) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        for (k, v) in value.graph.iter() {
            let mut attrs = vec![];
            if v.is_final {
                attrs.push(attr!("shape", "doublecircle"))
            }
            
            let node = node!(k, attrs);
            let node_edges: Vec<_> = v.transitions.iter().flat_map(|(symbol, states)| {
                let symbol_str = symbol.to_string();
                states.iter()
                    .map(|s| edge!(node_id!(k) => node_id!(s); attr!("label", symbol_str)))
                    .collect::<Vec<_>>()
            }).map(Stmt::from).collect();
            
            nodes.push(Stmt::from(node));
            edges.extend(node_edges);
        }
        
        let graph = Graph::DiGraph {
            id: id!("automaton"),
            strict: true,
            stmts: [nodes, edges].into_iter().flatten().collect(),
        };
        
        return graph;
    }
}


pub struct ExecutionContext<'a> {
    pub current_state: &'a State<'a>,
    pub automaton: &'a Automaton<'a>
}

impl<'a> ExecutionContext<'a> {
    pub fn new(automaton: &'a Automaton) -> Self {
        Self {
            current_state: &automaton.initial_state,
            automaton,
        }
    }
    
    pub fn run(&mut self, input: String) -> eyre::Result<bool> {
        for x in input.chars() {
            self.process(x)?;
        }
        
        Ok(self.current_state.is_final)
    }

    fn process(&mut self, input: char) -> eyre::Result<()> {
        let input_symbol = Symbol::from(input);
        let transitions = self.current_state.transitions.get(&input_symbol);

        let transitions = match transitions {
            None => bail!("Not found transition for symbol '{:?}' and state '{:?}'", input_symbol, self.current_state),
            Some(v) => v
        };

        let next_state = match transitions.first() {
            None => bail!("Not found next state for symbol '{:?}' and state '{:?}'", input_symbol, self.current_state),
            Some(nex_state) => nex_state
        };

        let next_state = match self.automaton.graph.get(&next_state.to_string()) {
            None => bail!("Not found next state '{}' for symbol '{:?}' and state '{:?}'", next_state, input_symbol, self.current_state),
            Some(nex_state) => nex_state
        };

        self.current_state = next_state;
        
        Ok(())
    }
}

fn input_line() -> io::Result<String> {
    let mut buf = String::new();

    io::stdin().read_line(&mut buf)?;

    Ok(buf)
}


fn main() -> eyre::Result<(), Report> {
    print!("Automaton file: ");
    
    let file = input_line()?;
    
    let automaton = Automaton::from_file(file)?;
    let mut evaluation = ExecutionContext::new(&automaton);
    
    print!("Input: ");
    
    let input = input_line()?;
    
    match evaluation.run(input) {
        Ok(result) => {
            if result {
                println!("Accepted")
            } else {
                println!("Rejected")
            }
        },
        Err(e) => {
            println!("Rejected: {}", e)
        }
    }
    
    Ok(())
}



#[cfg(test)]
mod tests {
    use crate::{Automaton, ExecutionContext};
    
    const EXAMPLE_AUTOMATA: &str = r#"q0
a b
q0 q1 q2 q3
q1 q3
q0 q1 a
q1 q1 a
q1 q2 b
q2 q1 b
q2 q3 a"#;

    #[test]
    pub fn parse_valid_automaton() {
        let input = EXAMPLE_AUTOMATA.clone();
        
        let actual = Automaton::from_text(input.clone());
        assert!(actual.is_ok());
        
        let actual = actual.unwrap();
    }
    
    #[test]
    pub fn run_valid_final() {
        let inputs = vec![
            "aa",
            "aaaaaaaa",
            "aaaba"
        ];

        let automaton = Automaton::from_text(EXAMPLE_AUTOMATA.clone())
            .expect("parse bug");
        
        for input in inputs {
            let mut context = ExecutionContext::new(&automaton);
            let actual = context.run(input.to_string());
            
            let actual = match actual {
                Ok(_) => actual,
                Err(e) => panic!("Should run without errors input {}: {}", input, e)
            };
            
            assert!(actual.unwrap(), "Should accept input {}", input)
        }
    }

    #[test]
    pub fn run_invalid_symbol() {
        let inputs = vec![
            "ac",
            "cc",
            "aaabc"
        ];

        let automaton = Automaton::from_text(EXAMPLE_AUTOMATA.clone())
            .expect("parse bug");

        for input in inputs {
            let mut context = ExecutionContext::new(&automaton);
            let actual = context.run(input.to_string());

            assert!(actual.is_err(), "Should run with error input {}", input);
        }
    }

    #[test]
    pub fn run_invalid_sequence() {
        let inputs = vec![
            "aaab"
        ];

        let automaton = Automaton::from_text(EXAMPLE_AUTOMATA.clone())
            .expect("parse bug");

        for input in inputs {
            let mut context = ExecutionContext::new(&automaton);
            let actual = context.run(input.to_string());

            let actual = match actual {
                Ok(_) => actual,
                Err(e) => panic!("Should run without errors input {}: {}", input, e)
            };

            assert!(!actual.unwrap(), "Should accept input {}", input)
        }
    }
}
