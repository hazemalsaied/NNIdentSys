### Se connecter à Grid
```
ssh halsaied@access.grid5000.fr
```
```
ssh nancy
```
### Lancer une expérimentation en mode passif:
```
oarsub -p "gpu<>'NO'" -q production -l nodes=1,walltime=5 /home/halsaied/NNIdenSys/Scripts/test-passive.sh -O out -E HUmlpErr
```
```
oarsub -p "cluster='graphique'" -q production -l nodes=1,walltime=5 /home/halsaied/NNIdenSys/Scripts/test-passive.sh -O out -E HUmlpErr
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
scp -r  halsaied@access.grid5000.fr:'/home/halsaied/nancy/mlpTotalErr' Reports
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