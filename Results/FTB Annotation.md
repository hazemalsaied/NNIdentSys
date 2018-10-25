# MWT (160 cas)
elle ajoute qu' il n' **a** jamais  **été**  **question**  **d'** attribuer à la chaîne culturelle européenne des fréquences " réservées au secteur privé " et présente 3 **au** **total** douze **points** **de** **désaccord** avec cette décision du csa .

### MWEs
1- oth: avoir

2- oth: être question de

3- oth: à total

4- oth: point de désaccord


### Cupt
1	Elle	il	CL	CLS	sentid=flmf7ag2ep-616|g=f|n=s|p=3|s=suj	2	suj	2	suj

2	ajoute	ajouter	V	V	m=ind|n=s|p=3|t=pst	0	root	0	root

3	qu'	que	C	CS	s=s	2	obj	2	obj

4	il	il	CL	CLS	g=m|n=s|p=3|s=suj	8	suj	8	suj

5	n'	ne	ADV	ADV	s=neg	8	mod	8	mod

6	a	avoir	V	V	m=ind|mwehead=V|n=s|p=3|t=pst	8	dep	8	dep

7	jamais	jamais	ADV	ADV	s=neg	8	mod	8	mod

8	été	être	V	VPP	m=part|mwehead=VPP|t=past	3	obj.cpl	3	obj.cpl

9	question	question	N	NC	g=f|n=s|s=c|component=y	8	dep_cpd	8	dep_cpd

10	d'	de	P	P	component=y	8	dep_cpd	8	dep_cpd

# Annotation bug dep_cpd without dep parent: line: 27260, word: bien DependencyParent: 24, head Position: 21
## selon le projet , les salariés qui sont mis en **congé** **-** **maladie** seront **ou** **bien** impayés le premier jour **ou** bien ils devront le déduire de leurs vacances .
### MWEs

1- oth: congé - maladie (+)

2- oth: ou bien (+)

3- oth: ou (+)

### Cupt

1	Selon	selon	P	P	sentid=flmf7ag2ep-771	14	mod	14	mod

2	le	le	D	DET	g=m|n=s|s=def	3	det	3	det

3	projet	projet	N	NC	g=m|n=s|s=c	1	obj.p	1	obj.p

4	,	,	PONCT	PONCT	s=w	14	ponct	14	ponct

5	les	le	D	DET	g=m|n=p|s=def	6	det	6	det

6	salariés	salarié	N	NC	g=m|n=p|s=c	14	suj	14	suj

7	qui	qui	PRO	PROREL	g=m|n=p|p=3|s=rel	9	suj	9	suj

8	sont	être	V	V	m=ind|n=p|p=3|t=pst	9	aux.pass	9	aux.pass

9	mis	mettre	V	VPP	g=m|m=part|n=s|t=past	6	mod.rel	6	mod.rel

10	en	en	P	P	_	9	mod	9	mod

11	congé	congé	N	NC	g=m|mwehead=NC|n=s|s=c	10	obj.p	10	obj.p

12	-	-	PONCT	PONCT	s=w|component=y	11	dep_cpd	11	dep_cpd

13	maladie	maladie	N	NC	g=f|n=s|s=c|component=y	11	dep_cpd	11	dep_cpd

14	seront	être	V	V	m=ind|n=p|p=3|t=fut	0	root	0	root

15	ou	ou	C	CC	mwehead=CC|s=c	14	coord	14	coord

16	bien	bien	ADV	ADV	component=y	15	dep_cpd	15	dep_cpd

17	impayés	impayés	V	VPP	g=m|m=part|n=p|t=past	15	dep	15	dep

18	le	le	D	DET	g=m|n=s|s=def	20	det	20	det

19	premier	premier	A	ADJ	g=m|n=s|s=ord	20	mod	20	mod

20	jour	jour	N	NC	g=m|n=s|s=c	17	mod	17	mod

21	ou	ou	C	CC	mwehead=CC|s=c	14	coord	14	coord

22	bien	bien	ADV	ADV	component=y	24	dep_cpd	**24**	dep_cpd

23	ils	il	CL	CLS	g=m|n=p|p=3|s=suj	24	suj	24	suj

24	**devront**	devoir	V	V	m=ind|n=p|p=3|t=fut	21	dep	21	dep

25	le	le	CL	CLO	g=m|n=s|p=3|s=obj	26	obj	26	obj

26	déduire	déduire	V	VINF	m=inf	24	obj	24	obj

27	de	de	P	P	_	26	mod	26	mod

28	leurs	son	D	DET	g=f|n=p|p=3|s=poss	29	det	29	det

29	vacances	vacances	N	NC	g=f|n=p|s=c	27	obj.p	27	obj.p

30	.	.	PONCT	PONCT	s=s	14	ponct	14	ponct
