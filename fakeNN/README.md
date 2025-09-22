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
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN-no-padding-no-db.csv no status no export &> 40x120x20wm-n-k-fakeNN-no-pad-no-db.output &
```



```
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN-no-padding-no-db.csv no status run no &> 40x120x20wm-n-k-fakeNN-no-pad-no-db.output &
```
```
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN-no-padding-no-db.csv no status no no &> 40x120x20wm-n-k-fakeNN-no-pad-no-db.output &
```
```
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN-no-padding-no-db.csv no compile no no &> 40x120x20wm-n-k-fakeNN-no-pad-no-db.output &
```
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
```
nohup bash startNoHangUpRun.sh 40x120x20wm-n-k-fakeNN.csv genJsons no no no &> 40x120x20wm-n-k-fakeNN.output &

```
```
nohup bash startNoHangUpRun.sh 40x120x20w7-24-10-fakeNN.csv genJsons compile no no skip &> 40x120x20w7-24-10-fakeNN.output &
```
```
nohup bash startNoHangUpRun.sh 251x500x600w8-24-8-fakeNN.csv genJsons compile no no skip &> 251x500x600w8-24-8-fakeNN.output &
```
```
nohup bash startNoHangUpRun.sh 8x672x672wm-n-k-fakeNN.csv genJsons compile no no skip &> 8x672x672wm-n-k-fakeNN.output &
```
```
nohup bash startNoHangUpRun.sh 56x56x56wm-n-k-fakeNN.csv genJsons compile no no &> 56x56x56wm-n-k-fakeNN.output &
```
```
nohup bash startNoHangUpRun.sh 120x40x20wm-n-k-fakeNN.csv genJsons compile no no &> 120x40x20wm-n-k-fakeNN.output &
```

### Converting Myrtle Search Spaces to FakeNN input format
```
python3 convertSSToFakeNNInput.py 40x120x20wm-n-k_case1_searchSpace.csv
```
```
python3 convertSSToFakeNNInput.py 8x672x672wm-n-k_searchSpace.csv
```
```
python3 convertSSToFakeNNInput.py 56x56x56wm-n-k_searchSpace.csv
```
```
python3 convertSSToFakeNNInput.py 120x40x20wm-n-k_searchSpace.csv
```