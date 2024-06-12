

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;
use std::io::BufRead;
use rand::Rng;
use serde::Serialize;
use serde_json;

use serde_json::Value;
use anyhow::Error;
use std::thread::available_parallelism;
use clap::Parser;
use std::path::PathBuf;
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use tokenizers::tokenizer::{
    Tokenizer
};

pub mod s3;
pub mod io;


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



/*=================================================================
=                             STAT COLLECTOR                      =
=================================================================*/

fn collect_path_counts(path: &PathBuf) -> Result<(PathBuf, usize, usize), Error> {
    // Given input pathbuf returns (path, total_documents, total_tokens)
    let mut full_token_count = 0;
    let mut docs_seen = 0;
    let tokenizer = Tokenizer::from_pretrained("EleutherAI/gpt-neox-20b", None).unwrap();

    let contents = read_pathbuf_to_mem(path).unwrap();

    for line in contents.lines() {
        let line = line.unwrap();
        let json: Value = serde_json::from_str(&line).unwrap();
        let text = json["text"].as_str().unwrap();
        let encoded = tokenizer.encode(text, false).unwrap();
        let token_length = encoded.get_ids().to_vec().len();
        full_token_count += token_length;
        docs_seen += 1;
    }

    Ok((path.clone(), docs_seen, full_token_count))
}


/*=================================================================
=                                 MAIN                            =
=================================================================*/

#[derive(Serialize)]
struct StatsResult {
    stats : Vec<(PathBuf, usize, usize)>
}

fn main() {
    let start_main = Instant::now();
    let args = ArgParser::parse();


    let paths = expand_dirs(args.input.clone(), None).unwrap();
    let pbar = build_pbar(paths.len(), "Paths");

    let outputs : Vec<(PathBuf, usize, usize)> = paths.par_iter()
        .map(|p| {
            let result = collect_path_counts(p).unwrap();
            pbar.inc(1);
            result 
        })
        .collect();

    let result = StatsResult { stats: outputs };
    let json_bytes: Vec<u8> = serde_json::to_vec(&result).unwrap();
    write_mem_to_pathbuf(&json_bytes, &args.output).unwrap();


    println!("-------------------------");
    println!("Finishing stats collection in {:?} seconds", start_main.elapsed().as_secs());

}
