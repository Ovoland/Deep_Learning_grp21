# Commandes pour submit un job sur RunAI 

.

.

.



## Submit un job à travers un interactif environment
Il faut reBUILD une docker image et CHANGER LE NOM DE LA TOOLBOX !!!! Très important ! C'est pour ça que ça marchait pas. Mettre uniquement transformers dans le requirements.txt sans spécifier une version. L'image se build et se push rapidement par rapport à nos anciens essaies.

### Ouvrir un environement interactif
Pour ouvrir l'environement interactif directement depuis le node de RunAI il faut run la commande qui suit en changeant UNIQUEMENT "username" ou le nom de la toolbox que vous avez créée si ce n'est pas déjà fait.
```bash
runai submit --image registry.rcp.epfl.ch/ee-559-username/new_toolbox:v0.1 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home --interactive --attach
```
Version de Thomas pour utiliser le meilleur gpu performance et perdre moins de temps
```bash
runai submit --image registry.rcp.epfl.ch/ee-559-serillon/new_toolbox:v0.1 --gpu 1 --node-pool a100-40g --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home --interactive --attach
```

```bash
runai submit --image registry.rcp.epfl.ch/ee-559-serillon/new_toolbox:v0.1 --gpu 1 --node-pool h100 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home --interactive --attach
```

Une fois ouvert, cet environement est un bash shell classique mais qui est directement sur le RunAI. IL NE LANCE PAS le "main.py" c'est normal. Avant de faire ça, vérifiez que le module "transformers" a bel et bien été importé en ouvrant une shell python:
```bash
python
```

.

.

.


### Vérifier que transformers a été correctement installé
Une fois le python shell ouvert, faites:
```python
>>> import transformers
```
Si ça produit une erreur, il y a un problème avec la docker image. Dans ce cas, il faut rebuild une docker image en mettant UNIQUEMENT transformers dans les requirements sans aucune version spécifique. Ou alors comme solution momentanée vous pouvez faire:
```python
>>> pip install transformers
```
Cependant ce n'est pas une solution très propre. Mieux vaut re-build la dockerimage.


Si ça ne produit aucune erreur, SUPER ! Tu peux fermer le python shell.
```python
>>> exit()
```

.

.

.


### Lancer le job / fichier main.py
Maintenant il faut lancer le fichier main.py. Il faut donc se déplacer dans le dossier correspondant. Comme la fonction pour ourvrir environement interactif fait un pcv vers un nouveau dossier (--pvc home:/pvc/home), il faut se déplacer dedans pour retrouver tous nos fichiers.
```bash
cd /pvc/home/Deep_Learning_grp21/               # pour Jeremy
cd /pvc/home/mini_project/Deep_Learning_grp21/  # pour Océane
cd ... ?                                        # pour Thomas
```

Ensuite tu peux directement lancer ton main !
```bash
python main.py
```

.

.

.

### Fermer l'environement interactif
```bash
exit
```


.

.

.

.
 
.

## Autres
### Ouvrir un environment intéractif avec la base dockerimage "pytorch:1.0.6" (dans le cas ou la méthode dessus ne marche pas)
```bash
runai submit --run-as-user --image nvcr.io/nvidia/ai-workbench/pytorch:1.0.6 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home --interactive --attach
```
Pour mettre transformers indépendament de la dockerimage:
```python
>>> pip install transformers
```

.

.

.

### Ancien fichier "runai_submit_jobs.sh" de Jeremy
```bash
#runai submit --name test2 --image registry.rcp.epfl.ch/ee-559-serillon/my-toolbox:v0.1 --gpu 1 --node-pools v100 --pvc home:${HOME} -e HOME=${HOME} --command -- python3 ~/Deep_Learning_grp21/main.py
runai submit --name test2 --image registry.rcp.epfl.ch/ee-559-serillon/my-toolbox:v0.1 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home --command -- python3 /pvc/home/Deep_Learning_grp21/main.py

runai submit --image registry.rcp.epfl.ch/ee-559-serillon/new_toolbox:v0.1 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home --interactive --attach

runai submit  --run-as-user --image nvcr.io/nvidia/ai-workbench/pytorch:1.0.6 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home --interactive --attach
```