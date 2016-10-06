from flask import Flask, render_template, request, redirect
from numpy import dot,sqrt,load,array,argsort,sum,count_nonzero,any
#from pandas import read_csv
from requests import get
from time import sleep
from xmltodict import parse
from random import sample, choice
from dill import load as lode
import networkx as nx
from json import dumps
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler
#from pandas import read_csv
#from bs4 import UnicodeDammit

app = Flask(__name__)

app.vars={}

class catMechTransformer():
    def __init__(self):
        self.cats=83
        self.mechs=51
    
    def fit(self,X,y):
        return self
    
    def flatten(self,l):
        cats=[0 for i in range(self.cats)]
        for i in l[5]:
            cats[i] = 1
        mechs=[0 for i in range(self.mechs)]
        for i in l[6]:
            mechs[i] = 1
        return l[:5]+cats+mechs
    
    def transform(self,X):
        return [self.flatten(i) for i in X]

def category_and_mechanic_table(games):
    categories=set([])
    mechanics=set([])
    for game in games:
        [categories.add(i) for i in game[10]]
        [mechanics.add(i) for i in game[11]]
    categories=list(categories)
    mechanics=list(mechanics)
    outData=[]
    for game in games:
        gameData=[]
        #gameData.append(int(game[0]))
        #removed the first entry which is the name
        #+map(lambda x: x/60. if x<600 else 10.,game[4:7])
        gameData=gameData+game[3:4]+[game[4] if game[4]<10 else 10]+[game[7]/60]+game[8:10]
        for category in categories:
            if category in game[10]:
                gameData.append(1)
            else:
                gameData.append(0)
        for mechanic in mechanics:
            if mechanic in game[11]:
                gameData.append(1)
            else:
                gameData.append(0)
        outData.append(gameData)
    return categories, mechanics, outData

def tanimotoSimilarity(user1,user2): #tanimoto
    z=dot(user1,user2)
    a=float(z)/(2-z)
    if sum(user1)==0 or sum(user2)==0: a = 0
    return a

def GameTree(row):
    d={}
    for i in range(len(app.gameData)):
        if app.gameData[i][15] > 500:
            d[i] = app.gameData[i][1]
    out = {}
    row = app.transformer.transform([row])[0]
    hypeSims  = [tanimotoSimilarity(row, i) for i in app.gameNorm]
    out['A Hypothetical Game']= list(set([(list(hypeSims).index(sorted(hypeSims)[-i]),3) for i in range(0,6)]))[:5]
    for k in [i[0] for i in out['A Hypothetical Game']]:
        rk = list(app.gameRecs[k])
        out[k] = list(set([(rk.index(sorted(rk)[-i]),2) for i in range(0,6)]))[:5]
        for j in [i[0] for i in out[k]]:
            rj = list(app.gameRecs[j])
            out[j] = list(set([(rj.index(sorted(rj)[-i]),1) for i in range(0,6)]))[:5]
    nodes=out.keys()
    for i in out.values():
        nodes+=i
    nodeNamed=[]
    for i in list(set(nodes)):
        try:
            nodeNamed+=[d[i]]
        except:
            continue
    G=nx.Graph()
    G.add_nodes_from(nodeNamed, group = 1)
    G.add_node(u'A Hypothetical Game', group =2)
    for i in out.keys():
        for j in out[i]:
            if i !=u'A Hypothetical Game':
                G.add_edge(d[i],d[j[0]],weight=j[1])
            else:
                G.add_edge(i,d[j[0]],weight=j[1])
    return G, [i[0] for i in out[u'A Hypothetical Game']], [d[i[0]] for i in out[u'A Hypothetical Game']]

# loading in resources. I know pickles and .npy files are unsafe, but for testing the performance gains were worth it. gameList is a list of games, their BoardGamegGek id number, and name. a Matrix is a precomputed similarity matrix of user recommendations such that we can dot our vector of game ratings into it to get a user-user similarity. aMatrixMasked is set up to allow us to quickly sum the similarities. It's the previous matrix with the nonzero entries set to one. It will allow us to quickly sum the similarities in user-user. This user-user uses cosine similarity.
#app.gamesNames=[[i[0],i[1]] for i in read_csv("gameList.csv", quotechar='"', skipinitialspace=True).as_matrix()]
app.recs=load('aMatrixHalfFloat.npy')
app.recsMasked = load('aMatrixMaskedHalfFloat.npy')
# itemitem is a similarity between the columns of a using Tanimoto similarity
app.itemRecs=load('itemitemHalfFloat.npy')
# itemitem is a similarity between the rows of a matrix of game information using Tanimoto similarity. it's slightly weighted towards game ownership and comments on the game.
app.gameRecs=load('gamegameHalfFloat.npy')
app.gameData=load('gameData.p')
app.gamesNames=[[i[0],i[1]] for i in app.gameData]
app.gameScores = [i[13] for i in app.gameData]
app.cats, app.mechs, app.gameNorm = category_and_mechanic_table(app.gameData)

app.transformer = StandardScaler().fit(app.gameNorm)
cNorm=[]
outData = app.transformer.transform(app.gameNorm)
for i in app.gameNorm:
	j=1./sqrt(float(dot(i,i)))
	cNorm.append(map(lambda x: j*x,i))
app.gameNorm = cNorm

#app.gamesNames=read_csv('gameList.csv',sep=r';',header=0).as_matrix()
#print app.gamesNames2[100]

app.model=lode(open('gamescoremodel','rb'))

@app.route('/')
def main():
	return render_template('landing.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/user',methods=['GET','POST']) 
def user():
	if request.method == 'GET':
		return render_template('userinfo.html')
	else:
		#request was a POST
		#grab the recommendation method
		app.vars['method']=int(request.form['method'])
		if  request.form['username']=='':
			#if there's no username grab all the games, filter them and grab the ratings
			t=request.form.getlist('game[]', type=float)
			print t
			app.vars['games'] = map(lambda x: int(x),filter(lambda x: x>-1,t))
			r=request.form.getlist('rating[]', type=float)
			app.vars['ratings'] = [r[i] for i in range(len(app.vars['games'])) if t>-1]
			#if there's no rating, pick a random 5 games and run on that
			if app.vars['games']==[]:
				#app.vars['games']=[355, 1050, 937, 569, 1051]
				app.vars['games']=sample(range(1189),5)
				app.vars['ratings']=[choice(range(5,11)) for i in range(5)]
# 			print app.vars['games']
# 			print app.vars['ratings']
			#some errors
			if (len(app.vars['games'])<3):
				return render_template('error.html',message='Please add more games!')
			elif  (len(set(app.vars['games']))<3):
				return render_template('error.html',message='Please rate unique games!')
			elif (count_nonzero(app.vars['ratings'])<3):
				return render_template('error.html',message='Please rate more games!')
			else:
				return redirect('/main_entered')
		else:
			#read a username and go to that branch
			app.vars["username"]=request.form['username']
			return redirect('/main_username')
			
			
			
			
@app.route('/game',methods=['GET','POST'])
def game():
	if request.method == 'GET':
		return render_template('gameinfo.html')
	else:
		#print 4
		try:
			app.vars['gameParts'] = [float(request.form['minplayers']), float(request.form['maxplayers']), float(request.form['avgplaytime']), float(request.form['langcomplexity']), float(request.form['playerage']), map(int,request.form.getlist('theme')), map(int,request.form.getlist('mechanics'))]
			score = app.model.predict([app.vars['gameParts']])[0]
			percentile = 100 * round( float(len([i for i in app.gameScores if i <= score ])) / len(app.gameScores),4)
			row = catMechTransformer().transform([app.vars['gameParts']])[0]
			a, b, c = GameTree(row)
			return render_template('gameresults.html',score=str(score), percentile=str(percentile)+'%', games=c, nums=b, game_text=[app.gameData[i][-1] for i in b], game_json=dumps(json_graph.node_link_data(a))) # app.vars['gameParts'][0])
		except:
			return render_template('error.html',message='Please enter valid input.')




#the string that appears if a username is invalid
noUser =  u'<?xml version="1.0" encoding="utf-8" standalone="yes" ?>\n<errors>\n\t<error>\n\t\t<message>Invalid username specified</message>\n\t</error>\n</errors>'

@app.route('/main_username',methods=['GET','POST'])
def main_username():
	#grab the XML
	XML=makeRequest(app.vars["username"])
	if XML==noUser:
		return render_template('error.html',message='No one exists with that username!')
	#parse it. I use the xmltodict library, which makes the output look like JSON data.
	XML=parse(XML)
	#print XML
	print XML['items']['@totalitems']
	try:
		#some users have data that's slightly misformatted. This catches those entries
		if int(XML['items']['@totalitems'])<3:
			return render_template('error.html',message='Please rate more games')
	except: 
		pass
	#a badly named function that grabs the names and ratings from the XML
	toGameAndRating(XML)
	del XML
	#catches some errors
	if (len(app.vars['games'])<4) or (count_nonzero(app.vars['ratings'])<5):
		return render_template('error.html',message='Please rate more games.')
	#return render_template('error.html',message=app.gamesNames[2][1])
	return redirect('/main_entered')

@app.route('/main_entered',methods=['GET','POST'])
def main_entered():
	#takes our short list of games and returns a vector that's the height of our matrices and initializes some vectors
	game_row=[0 for i in range(len(app.gamesNames))]
	rating_row=[0 for i in range(len(app.gamesNames))]
	owner_suggestions=[]
	rating_suggestions=[]
	game_suggestions=[]
	for i in range(len(app.vars['games'])):
		game_row[int(app.vars['games'][i])]=1
		rating_row[int(app.vars['games'][i])]=float(app.vars['ratings'][i])
	#print game_row[:26]
	owner_suggestions=ownership_recs(rating_row,app.itemRecs)
	rating_suggestions=getRecs(rating_row)
	game_suggestions=ownership_recs(rating_row,app.gameRecs)
	#a function that takes the output and mixes it together, removing the initial games as well.
	ratings,keys = copacetic(owner_suggestions,rating_suggestions,game_suggestions)
	print keys
	inProps,outProps=properties(keys)
	#print [UnicodeDammit(i[2]).unicode_markup for i in ratings]
	#=[int(i[1]) for i in ratings]
	#return render_template('error.html',message=UnicodeDammit(ratings[1][2]).unicode_markup)
	#print [i[3].replace('\,',',').replace('\n\n','</p><p>') for i in ratings]
	return render_template('results2.html',sim=[i[0] for i in ratings],games=[i[2].replace('\,',',') for i in ratings],nums=[int(i[1]) for i in ratings],inprops=inProps, outprops = outProps, game_text = [i[3].replace('\,',',').replace('\n\n','</p><p>') for i in ratings])#inProps,outprops=outProps)

#the gameData table has a bunch of comments about each game, this function pulls out the most common.
def properties(keys):
	print keys
	inpropsMech,inpropsCat=[],[]
	outpropsMech,outpropsCat=[],[]
	for i in app.vars['games']:
		inpropsMech=inpropsMech+app.gameData[i][11]
		inpropsCat=inpropsCat+app.gameData[i][10]
	for i in keys:
		outpropsMech=outpropsMech+app.gameData[i][11]
		outpropsCat=outpropsCat+app.gameData[i][10]
	def mostCommon(lst):
		def capFix(w):
			return w.lower() if (("War" not in w or w!='Wargames') and ('Nap'not in w) and ('Ren' not in w) and ('Arab' not in w)) else w
		return sorted(set(map(capFix,lst)), key=lst.count)[::-1][:3]
	print [mostCommon(inpropsCat),mostCommon(inpropsMech)],[mostCommon(outpropsCat),mostCommon(outpropsMech)]
	return [mostCommon(inpropsCat),mostCommon(inpropsMech)],[mostCommon(outpropsCat),mostCommon(outpropsMech)]

def copacetic(owner,rating,game):
	if app.vars['method'] == 1:
		new = rating
	elif app.vars['method']==2:
		new = game
	elif app.vars['method']==3:
		new = owner
	else:
		#this mixing seems to give a reasonably large number of games in the top 10.
		new = (0.5*array(owner/max(owner)))+(0.25*array(rating/max(rating)))+(0.25*array(game/max(game)))
		print max(new)
	if app.vars['games']!=[]:
		print 'copacetic', app.vars['games'],new[:10]
		for i in app.vars['games']:
			new[i] = 0
	#normalize the entries
	new=new/max(new)
	newkeys=sorted(range(len(new)), key=lambda x: new[x])[-50:] #50 largest simularities
	names=[]
	a=[]
	#removes some versions of the same game from the results
	newkeys.reverse()
	newnewkeys=[]
	for key in newkeys:
		if (app.gamesNames[key][1].split(':'))[0] not in names:
			a+=[[new[key],app.gamesNames[key][0],app.gamesNames[key][1],app.gameData[key][-1]]] 
			names+=[(app.gamesNames[key][1].split(':')[0])]
			newnewkeys.append(key)
	return a[:10],newnewkeys[:10]
		
#calculating the different sorts of recommendations

def ownership_recs(user,recs):
	totals=dot(user,recs)
	simssum=dot(map(lambda x: 1 if x>0 else 0,user),recs)
	rankings=[app.gameData[i][13] + totals[i]/simssum[i] for i in range(len(totals))]
	del totals,simssum
	#print 'item rankings', rankings[:20]
	return rankings


#if the rows are normalized cosine similarity reduces to a dot product between the matrix we care about and our vector of ratings. The normalization of the vector will eventually cancel out so we don't have to worry about it

def getRecs(user):
    #sims=[similarity(user,i) for i in recs]
    totals = dot(user,app.recs)
    simssum= dot(user,app.recsMasked)
    rankings=[totals[i]/simssum[i] for i in range(len(totals))]
    del totals,simssum
    #print 'rankings', rankings[:20]
    return rankings

#these functions handle data coming from a username request

def toRow(tups, index):
	out = [0 for i in range(len(app.gamesNames))]
	for tup in tups:
		while True:
			try:
				i=(app.gamesNames).index(tup[0])
				out[i]=tup[index] if tup[index]!=None else 0
				break
			except ValueError:
				break
	return out
       
def toGameAndRating(xml):
	gameids,ratings,owned=[],[],[]
	intGameList=[int(i[0]) for i in app.gamesNames]
	for i in xml[u'items'][u'item']:
		w = 0 if i[u'stats'][u'@numowned']== u'' else i[u'stats'][u'@numowned']
		if (int(w)>3000) and (i[u'@subtype']==u'boardgame') and int(i[u'@objectid']) in intGameList:
			gameids.append(int(i[u'@objectid']))
			ratings.append(0 if (i[u'stats'][u'rating'][u'@value']== u'N/A') else float(i[u'stats'][u'rating'][u'@value']))
			owned.append(returnOwnership(i))
	gameids,ratings,owned=array(gameids),array(ratings),array(owned)
	ids=gameids.argsort()
	gameids,ratings,owned=gameids[ids],ratings[ids],owned[ids]
	gameids=map(lambda x: intGameList.index(x),gameids)
	app.vars['games']=gameids
	app.vars['ratings']=ratings
	print gameids
	print list(ratings)

def removeDups(ratings):
    last=ratings[0]
    tups=[]
    for rating in ratings:
        if rating[0]!=last[0]:
            tups.append(last)
        last=rating
    tups.append(last)
    return tups
    
errorMessage=u'<html>\n  <head>\n    <title>Error 503 Service Unavailable</title>\n  </head>\n  <body>\n    <h1>503 Service Unavailable</h1>\n    Our apologies for the temporary inconvenience. The requested URL generated 503 "Service Unavailable" error due to overloading or maintenance of the server.\n   </body>\n</html>\n'

def makeRequest(name):
	r = get("https://www.boardgamegeek.com/xmlapi2/collection?username={}&stats=1".format(name))
	sleep(1)
	while True:
		try:
			r = get("https://www.boardgamegeek.com/xmlapi2/collection?username={}&stats=1".format(name))
			#print len(r.text)
			r=r.text
			if r==errorMessage: #we've overloaded the server
				sleep(30)
			elif (len(r)>188): #the waiting message is 187 characters, the error message is 300
				return r 
			elif len(r)==187:
				sleep(0.15)
				names.append(name)
			else:
				return r
		except requests.ConnectionError:
			sleep(0.15)
			
def returnOwnership(item):
    ownership= 1 if (item[u'status'][u'@own']=='1')or(item[u'status'][u'@prevowned']=='1')or(item[u'status'][u'@preordered']=='1') else 0
    wanting = 0.5 if  (item[u'status'][u'@want']=='1')or(item[u'status'][u'@wanttobuy']=='1')or(item[u'status'][u'@wishlist']=='1') else 0
    return ownership #max([ownership,wanting]) 

@app.errorhandler(404)
def error(e):
	return render_template('error.html',message='Something\'s wrong.')
	
@app.errorhandler(500)
def error(e):
	return render_template('error.html',message='Something\'s wrong.')

if __name__ == '__main__':
	app.debug = False
#	app.run(port=33507)
	app.run(port=33508)
#	app.run(host='0.0.0.0')