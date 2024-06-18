

use std::collections::HashMap;
use unicode_segmentation::UnicodeSegmentation;


use std::time::Instant;
use std::io::BufRead;

use serde::Serialize;
use serde_json;

use serde_json::Value;
use anyhow::Error;

use clap::Parser;
use std::path::PathBuf;
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use dashmap::DashMap;

pub mod s3;
pub mod io;

const DELVE: [&str; 8] = ["delve", "delves", "delved", "delving", "Delve", "Delves", "Delved", "Delving"];

/*=================================================================
=                                  ARGS                           =
=================================================================*/



#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[arg(long, required=true, num_args=1..)]
    input: Vec<PathBuf>,

    #[arg(long, required=true)]
    output: PathBuf,

    #[arg(long, default_value_t=1_000_000)]
    reservoir_size: usize,

    #[arg(long, default_value_t=0)]
    threads: usize,
}


/*=================================================================
=                             UTILITIES                           =
=================================================================*/


fn build_pbar(num_items: usize, units: &str) -> ProgressBar {
    let mut template = String::from(units);
    template.push_str(" {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]");
    let pbar = ProgressBar::new(num_items as u64)
        .with_style(
            ProgressStyle::with_template(&template).unwrap()
        );

    pbar.inc(0);
    pbar
}


fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split_word_bounds().filter(|w| {
        for c in w.chars() {
            if !c.is_whitespace() {
                return true;
            }
        }
        false
    })
}


/*=================================================================
=                             STAT COLLECTOR                      =
=================================================================*/

fn collect_path_counts(path: &PathBuf, delve_counter: &DashMap<String, usize>) -> Result<(usize, usize), Error> {
    // Given input pathbuf returns (path, total_documents, total_tokens)
    let mut full_byte_length = 0;
    let mut docs_seen = 0;

    let contents = read_pathbuf_to_mem(path).unwrap();
    for line in contents.lines() {
        let line = line.unwrap();
        let json: Value = serde_json::from_str(&line).unwrap();
        let text = json["text"].as_str().unwrap();
        full_byte_length += text.len();
        for token in tokenize(text) {
            let token = token.trim();

            for allomorph in DELVE {
                if allomorph == token {
                    delve_counter.entry(allomorph.to_string()).or_insert(0);
                    delve_counter.alter(allomorph, |_, count| {
                        count + 1
                    })
                }
            }

        }
        docs_seen += 1;
    }

    Ok((docs_seen, full_byte_length))
}


/*=================================================================
=                                 MAIN                            =
=================================================================*/

#[derive(Serialize)]
struct StatsResult {
    stats : Vec<(usize, usize)>,
    delve_counter: HashMap<String, usize>

}

fn main() {
    let start_main = Instant::now();
    let args = ArgParser::parse();


    let paths = expand_dirs(args.input.clone(), None).unwrap();
    let pbar = build_pbar(paths.len(), "Paths");

    let delve_counter: DashMap<String, usize> = DashMap::new();
    let outputs : Vec<(usize, usize)> = paths.par_iter()
        .map(|p| {
            let result = collect_path_counts(p, &delve_counter).unwrap();
            pbar.inc(1);
            result 
        })
        .collect();

    let delve_counter_hmap: HashMap<String, usize> = delve_counter.clone().into_iter().collect();        
    let total_docs : usize = outputs.iter().map(|(d, _)| d).sum();
    let total_bytes : usize = outputs.iter().map(|(_,b)| b).sum();

    let result = StatsResult { stats: outputs , delve_counter: delve_counter_hmap};
    let json_bytes: Vec<u8> = serde_json::to_vec(&result).unwrap();
    write_mem_to_pathbuf(&json_bytes, &args.output).unwrap();



    println!("-------------------------");
    println!("Saw {:?} bytes of data", total_bytes);
    println!("Saw {:?} docs", total_docs);
    println!("Delve counts {:?}", delve_counter);
    println!("Finishing stats collection in {:?} seconds", start_main.elapsed().as_secs());

}
