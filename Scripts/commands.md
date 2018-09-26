### Se connecter à Grid
```
ssh halsaied@access.grid5000.fr
```
```
ssh nancy
```
### Lancer une expérimentation en mode passif:
```
oarsub -p "gpu<>'NO'" -q production -l nodes=1,walltime=50 /home/halsaied/NNIdenSys/Scripts/test-passive.sh -O out -E HUmlpErr
```
```
oarsub -p "cluster='graphique'" -q production -l nodes=1,walltime=5 /home/halsaied/NNIdenSys/Scripts/test-passive.sh -n HUmlpErr -O out -E HUmlpErr
```
```
env MKL_THREADING_LAYER=GNU  python NNIdenSys/src/xpNonCompo.py
```
```
oarsub -p "gpu<>'NO'" -q production -l nodes=1,walltime=50 /home/halsaied/NNIdenSys/Scripts/attached-test-passive.sh -O out -E Reports/3-noPaddingPOSToken
oarsub -p "gpu<>'NO'" -q production -l nodes=1,walltime=75 /home/halsaied/NNIdenSys/Scripts/test-passive.sh -n token.pos.noPadding -O 5.token.pos.noPadding -E 5.token.pos.noPadding.err
```
##Activer l'enveronnement virtuel
```
source miniconda2/bin/activate
```
#Lancer une expérimentation en mode Interactif:
```
oarsub -p "gpu<>'NO'" -l nodes=1,walltime=5 -q production -I
```
```
oarsub -p "cluster='graphique'"  -l nodes=1,walltime=5 -q production -I
```
```
sh NNIdenSys/Scripts/test-interactive.sh
```
### Update the code
```
scp -r  src halsaied@access.grid5000.fr:/home/halsaied/nancy/NNIdenSys
```
### Download a report
```
scp -r  halsaied@access.grid5000.fr:'/home/halsaied/nancy/Reports' Reports
```
### Update the scripts
```
scp -r  Scripts halsaied@access.grid5000.fr:/home/halsaied/nancy/NNIdenSys
```
### Update the whole project
```
scp -r  NNIdenSys halsaied@access.grid5000.fr:/home/halsaied/
```
### Supprimer un dossier
```
rm -rf NNIdenSys/Reports
```
### Créer un dossier
```
mkdir /home/halsaied/NNIdenSys/Reports
```
### Vider les caches de Theano
``` 
rm -rf .theano
```
```
chmod +x /home/halsaied/NNIdenSys/Scripts/deep-test-passive.sh
```


oarsub -p "gpu<>'NO'" -q production -l nodes=1,walltime=20 "NNIdenSys/Scripts/nonCompo.sh xpLbl=learning2" -n learning2 -O Reports/learning2 -E Reports/err

### Mode interactive
```
pLbl=xp8  NNIdenSys/Scripts/nonCompoI.sh
```

### Cudnn version
```
nvcc --version
```
### Kiperwasser
```
oarsub -p "cluster='grimani'" -q production -l nodes=1,walltime=3 -I
```

```
python3 evaluate.py --gold ST2/RNN/FR.gold.txt --pred ST2/RNN/FR.txt --train ../ressources/sharedtask.2/FR/train.cupt
```