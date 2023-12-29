import argparse
from diffusion_kinetics.pipeline.single_pipeline import SinglePipeline
from diffusion_kinetics.pipeline.multi_pipeline import MultiPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Kinetics')
    parser.add_argument('-i', '--input', required=True, help='Input file')
    parser.add_argument('-c', '--config', required=False, help='Config file')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        pipeline = MultiPipeline(args.input, args.output)
        print("Running multi-process pipeline...")
        pipeline.run(args.config)
    except:
        pipeline = SinglePipeline(args.input, args.output)
        print("Running single-process pipeline...")
        pipeline.run(args.config)
    
if __name__ == "__main__":
    main()