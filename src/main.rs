

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

fn collect_stats(chunk: &Vec<PathBuf>, reservoir_size: usize, thread_num:usize, pbar: Arc<Mutex<ProgressBar>>) -> Result<(Vec<usize>, usize, usize), Error> {
    let mut reservoir: Vec<usize> = Vec::new();
    let mut full_token_count = 0;
    let mut docs_seen = 0;
    let tokenizer = Tokenizer::from_pretrained("EleutherAI/gpt-neox-20b", None).unwrap();
    let mut rng = rand::thread_rng();
    for path in chunk {
        let contents = read_pathbuf_to_mem(path).unwrap();
        for line in contents.lines() {
            let line = line.unwrap();
            let json: Value = serde_json::from_str(&line).unwrap();
            let text = json["text"].as_str().unwrap();
            let encoded = tokenizer.encode(text, false).unwrap();
            let token_length = encoded.get_ids().to_vec().len();

            if docs_seen < reservoir_size {
                reservoir.push(token_length);
            } else {
                let j = rng.gen_range(0..=docs_seen);
                if j < reservoir_size {
                    reservoir[j] = token_length;
                }
            }
            full_token_count += token_length;
            docs_seen += 1;
           }
        pbar.lock().unwrap().inc(1);
    }

    Ok((reservoir, full_token_count, docs_seen))
}


/*=================================================================
=                                 MAIN                            =
=================================================================*/

#[derive(Serialize)]
struct StatsResult {
    total_tokens: usize,
    total_docs: usize,
    avg_tokens: f64,
    median_tokens: usize
}

fn main() {
    let start_main = Instant::now();
    let args = ArgParser::parse();


    let paths = expand_dirs(args.input.clone(), None).unwrap();
    let total_paths = paths.len();
    let pbar = build_pbar(paths.len(), "Paths");
    let pbar = Arc::new(Mutex::new(pbar));
    let threads = available_parallelism().unwrap().get();
    let chunk_size = (paths.len() + threads -1) / threads;
    let chunks: Vec<Vec<PathBuf>> = paths.chunks(chunk_size).map(|chunk| chunk.to_vec()).collect();
    let outputs: Vec<(Vec<usize>, usize, usize)> = chunks
        .par_iter()
        .enumerate()
        .map(|(thread_num, chunk)| collect_stats(chunk, args.reservoir_size, thread_num, pbar.clone()).unwrap())
        .collect();


    let total_tokens = outputs.iter().map(|(_, t,_)| t).sum::<usize>();
    let total_docs = outputs.iter().map(|(_, _, d)| d).sum::<usize>();

    let mut all_reservoirs: Vec<usize> = outputs.into_iter().flat_map(|(v, _, _)| v).collect();
    all_reservoirs.par_sort();
    let total = all_reservoirs.iter().sum::<usize>();
    let avg_tokens = total as f64 / all_reservoirs.len() as f64;
    let median_tokens = all_reservoirs[all_reservoirs.len() / 2];


    let result = StatsResult { total_tokens, total_docs, avg_tokens, median_tokens};
    let json_bytes: Vec<u8> = serde_json::to_vec(&result).unwrap();
    write_mem_to_pathbuf(&json_bytes, &args.output).unwrap();


    println!("-------------------------");
    println!("Finishing stats collection in {:?} seconds", start_main.elapsed().as_secs());
    println!("Processed {:?} total tokens", total_tokens);
    println!("Processed {:?} total docs", total_docs);
    println!("Median token length is {:?}", median_tokens);
    println!("Average token length is {:?}", avg_tokens);
}
