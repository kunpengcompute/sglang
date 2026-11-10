"""Select exact request lengths; preserve the shared streaming benchmark client."""
import argparse
import bench
p = argparse.ArgumentParser()
p.add_argument('--input-tokens',type=int,required=True)
p.add_argument('--output-tokens',type=int,required=True)
p.add_argument('--concurrency',type=int,required=True)
p.add_argument('--port',type=int,required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
bench.CASES['requested']=(a.input_tokens,a.output_tokens,a.concurrency)
bench.main(['--case','requested','--port',str(a.port),'--output',a.output,'--timeout','1800','--seed','77101'])
