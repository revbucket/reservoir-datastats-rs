

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

fn collect_stats(paths: Arc<Mutex<Vec<PathBuf>>>, reservoir_size: usize, pbar: Arc<Mutex<ProgressBar>>) -> Result<(Vec<usize>, usize, usize), Error> {
    let mut reservoir: Vec<usize> = Vec::new();
    let mut docs_seen = 0;
    let tokenizer = Tokenizer::from_pretrained("EleutherAI/gpt-neox-20b", None).unwrap();
    let mut rng = rand::thread_rng();

    loop {
        let res = paths.lock().unwrap().pop().clone(); 
        if res.is_none() {
            break;
        }
        let path = res.unwrap();
        let contents = read_pathbuf_to_mem(&path).unwrap();
        for line in contents.lines() {
            let _ = line.unwrap();
            docs_seen += 1;
           }
        pbar.lock().unwrap().inc(1);
    }

    Ok((reservoir, 0, docs_seen))
}


/*=================================================================
=                                 MAIN                            =
=================================================================*/

#[derive(Serialize)]
struct StatsResult {
    total_docs: usize,
}

fn main() {
    let start_main = Instant::now();
    let args = ArgParser::parse();


    let paths = expand_dirs(args.input.clone(), None).unwrap();
    let pbar = build_pbar(paths.len(), "Paths");
    let pbar = Arc::new(Mutex::new(pbar));
    let num_threads = if args.threads == 0 {
        available_parallelism().unwrap().get()   
    } else {
        args.threads
    };

    let paths = Arc::new(Mutex::new(paths));
    let threads : Vec<usize> = (0..num_threads).collect();
    let outputs : Vec<(Vec<usize>, usize, usize)> = threads.par_iter()
        .map(|_| collect_stats(paths.clone(), args.reservoir_size, pbar.clone()).unwrap())
        .collect(); 


    let total_docs = outputs.iter().map(|(_, _, d)| d).sum::<usize>();



    let result = StatsResult { total_docs};
    let json_bytes: Vec<u8> = serde_json::to_vec(&result).unwrap();
    write_mem_to_pathbuf(&json_bytes, &args.output).unwrap();


    println!("-------------------------");
    println!("Finishing stats collection in {:?} seconds", start_main.elapsed().as_secs());
    println!("Processed {:?} total docs", total_docs);

}
