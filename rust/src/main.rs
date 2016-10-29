use std::collections::HashMap;
use std::env;

type State = usize;
struct Alphabet(char);

struct HiddenMarkovShape {
  alphabets: Vec<Alphabet>,
  states: Vec<State>,
}

struct HiddenMarkov {
  shape: HiddenMarkovShape,
  alphabetprobs: Vec<HashMap<Alphabet, f32>>,
  transprobs: Vec<Vec<f32>>,
}

impl HiddenMarkov {
  fn read() -> HiddenMarkov {
    unimplemented!()
  }
}

fn main() {
  let hm = HiddenMarkov::read();
  let args: Vec<String> = env::args().collect();
  match &*args[0] {
    "1" => { task1(&hm); },
    "2" => { task2(&hm); },
    "3" => { task3(&hm); },
    "4" => { task4(&hm); },
    _ => {
      task1(&hm);
      task2(&hm);
      task3(&hm);
      task4(&hm);
    },
  }
}

fn task1(hm: &HiddenMarkov) {
}

fn task2(hm: &HiddenMarkov) {
}

fn task3(hm: &HiddenMarkov) {
}

fn task4(hm: &HiddenMarkov) {
}
