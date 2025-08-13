# Time Single Dispatches
#### Don't forget to
```
source venv/bin/activate
```
#### without no hang up
```
. run_linear_layer.sh one-run.csv genJsons compile run no
```
#### with no hang up
```
nohup bash startNoHangUpRun.sh one-more-run.csv no no no export skip &> one-more-run.output &
```
```
nohup bash startNoHangUpRun.sh one-more-run.csv genJsons compile run no skip &> one-more-run.output &

```
```
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN.csv genJsons no no no skip &> 40x120x20wm-n-k-fakeNN.output &
```
```
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN.csv no compile no no &> 40x120x20wm-n-k-fakeNN.output &
```
```
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN.csv no status no no skip &> 40x120x20wm-n-k-fakeNN.output &
```
```
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN.csv no no run no &> 40x120x20wm-n-k-fakeNN.output &
```

### Converting Myrtle Search Spaces to FakeNN input format
```
python3 convertSSToFakeNNInput.py 40x120x20wm-n-k_case1_searchSpace.csv
```