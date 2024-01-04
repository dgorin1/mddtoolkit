from diffusion_kinetics.pipeline.multi_pipeline import MultiPipeline
from diffusion_kinetics.optimization import Dataset
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Kinetics')
    parser.add_argument('-i', '--input', required=True, help='Input file')
    parser.add_argument('-c', '--config', required=True, help='Config file')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    return parser.parse_args()

def main():
    args = parse_args()
    MultiPipeline(args.output).run(args.config, args.input)
    
if __name__ == "__main__":
    main()