import numpy as np
from matplotlib import pyplot as plt


# fitting set 1
# a = [3.479942935951654, 2.826489944107995, 2.821469334948218, 2.817186107605047, 2.8134397944721368, 2.8126690716616314, 2.812749921989239, 2.8127968024484225, 2.812814058050273, 2.812825078390301, 2.812827887042691, 2.812830170895908, 2.8128295445782077, 2.8128299130809644, 2.812830825901179, 2.8128288644477606, 2.812829212359876, 2.812830982975488, 2.8128302626447272, 2.812831057541562, 2.8128303157333687, 2.812830924009535, 2.8128289361497307, 2.812830323741969, 2.8128301537685236, 2.8128301186810214, 2.812830209901566, 2.8128306140452333, 2.812830933999406, 2.812830856360812, 2.812830003792214, 2.8128307135026027, 2.8128301644783735, 2.812830535456642, 2.812830922353368, 2.812830655624015, 2.812830652996746, 2.8128306439078106, 2.812830169264242, 2.812830175059902, 2.8128302291245677, 2.8128310102255525, 2.812830902184901, 2.8128300198047915, 2.812830909862764, 2.812830486368886, 2.81283086442118, 2.8128312751026003, 2.8128306739916784, 2.8128305826228357, 2.812830401964811, 2.8128306338207185]
# b = [3.479942935951654, 2.189609645051855, 2.1658065170413514, 2.1455414663764674, 2.1280404686520775, 2.1233938782361474, 2.1228725929478505, 2.1226815855675314, 2.1225941227884015, 2.122567631750482, 2.122548085039588, 2.1225456923501236, 2.1225351892673223, 2.1225321028009025, 2.1225357337971387, 2.1225223339004144, 2.122519919224925, 2.122531898977092, 2.1225294876560734, 2.122534941379674, 2.1225317431476354, 2.122535405984038, 2.122522473363119, 2.122528023647622, 2.1225332334473586, 2.1225310501795187, 2.1225302800565524, 2.1225325131750834, 2.1225354047066505, 2.122536094955869, 2.1225306992887787, 2.12253391016926, 2.122530673043156, 2.1225322899525656, 2.1225353149381214, 2.1225347287504452, 2.1225348234664074, 2.1225348383046514, 2.1225315354682426, 2.122530264636673, 2.122529763319962, 2.1225348638031845, 2.1225359507011876, 2.1225306317964105, 2.1225352194216125, 2.1225334734884536, 2.1225357795927016, 2.1225394483702225, 2.1225368111303298, 2.122535695064821, 2.122533837453723, 2.1225346040961433]
# with open('loss_values_fittingset1_points10.txt', 'w') as file:
#     file.write('Fitting loss: \n') 
#     for num, val in enumerate(a):
#         if num < 51:
#           file.write(str(val)+'\n')
#     file.write('Validation loss: \n') 
#     for num, val in enumerate(b):
#         if num < 51:
#           file.write(str(val)+'\n')
# plt.plot([i for i in range(50)], a[:50], color = 'r')
# plt.plot([i for i in range(50)], b[:50], color = 'b')
# plt.show()
# ## Simulation set
# a =[2.8486578762176125, 1.1285267835436426, 0.7352801818987595, 0.6200112273472491, 0.5734132027462505, 0.5475722081721749, 0.5316014429723627, 0.5214858199777698, 0.5151279926205473, 0.5111606923287848, 0.5086871304386225, 0.5070779740130199, 0.505932635605947, 0.5050283917270351, 0.5042438570418968, 0.5035250148716857, 0.5028335361471685, 0.5021633197050599, 0.5015004925294786, 0.5008478671445593, 0.5002045393196745, 0.49957043269208007, 0.4989484565612389, 0.4983355290076507, 0.4977299334282157, 0.49713928605693686, 0.4965589732006734, 0.4959895875871611, 0.49542878703703924, 0.49488346418415413, 0.49434558435276477, 0.49382252726481024, 0.49330974883136375, 0.49280706360547144, 0.49231537573278217, 0.4918338371667299, 0.4913607672552193, 0.4909013547010813, 0.49045133156797255, 0.4900108864527904, 0.4895779453394812, 0.48915603879381314, 0.48874611475876, 0.48834422262511656, 0.48795115618845547, 0.4875668135828156, 0.4871914106336021, 0.48682476238393224, 0.48646656887773554, 0.48611681345166186, 0.48577351707058114, 0.48543943450226373, 0.4851153933447098, 0.48479641043160715, 0.48448580128195884, 0.48418302933782054]
# b = [2.8486578762176125, 2.396900794120538, 2.2289123323315327, 2.166729384355537, 2.141802297142719, 2.1296252273760445, 2.121318868847106, 2.114208002916779, 2.1075022773565166, 2.101010441949435, 2.094666629570683, 2.0885478692116646, 2.082688626548231, 2.077111410395138, 2.071823191154827, 2.066839042609056, 2.0621441820047814, 2.057744325085821, 2.053618586169109, 2.049762944204128, 2.0461664791667227, 2.0428170928496305, 2.0397073356542994, 2.0368207776404033, 2.034138937105761, 2.031662246144897, 2.029373364558036, 2.0272606231222516, 2.0253076016570355, 2.0235164469921165, 2.0218629281539258, 2.020350257241623, 2.018964260189676, 2.017693147267062, 2.016532557037182, 2.0154712698602593, 2.014500656941441, 2.013621974397293, 2.0128248971536182, 2.0121024492207384, 2.011445122136918, 2.0108547256886418, 2.010330304278935, 2.009860651418436, 2.0094430287208067, 2.009072693910185, 2.0087478557856575, 2.008465565635472, 2.008221863460227, 2.0080148210967375, 2.007837949005387, 2.0076942281513968, 2.007584448707784, 2.0074971108323587, 2.0074353770853017, 2.0073982191548856]
# print(len(a),len(b))
# with open('loss_values_simulation_moderate_noise.txt', 'w') as file:
#     file.write('## Fitting loss: \n') 
#     for num, val in enumerate(a):
#         if num < 51:
#           file.write(str(val)+'\n')
#     file.write('\n\n\n## Validation loss: \n') 
#     for num, val in enumerate(b):
#         if num < 51:
#           file.write(str(val)+'\n')
# plt.plot([i for i in range(50)], a[:50])
# plt.plot([i for i in range(50)], b[:50])
# plt.show()

# idx = [[10, 3, 8, 5, 4, 7],[4, 5, 7, 2, 0, 3],
# [2,4,1,10,3,7], 
# [7, 8, 1, 9, 4, 10],
# [0, 5, 8, 4, 7, 1],
# [0, 5, 8, 4, 7, 1],
# ]
# M = [[1.227800695471329,1.563128913092063,1.2608017535588336,1.5938006657373742],
# [2.316277751278483,2.3242367616496598,2.3841142942407987,2.3822228086893635],
# [2.943713906669816,2.1887437438523514,2.9984283274414564,2.243317476648786],
# [3.632230497031364,1.5615890927255605,3.6930002667449227,1.599446059060236],
# [1.8804374221159864,1.2084766456065112,1.9221215910321467,1.23762687724457],
# [1.2759449447948583,1.1904389968311175,1.3082953357416833,1.2164116205545503],
#   ]

# S = [[0.08143014938517425,0.12841103511959623,0.07794326471906977,0.12595330295876966],
# [0.6367783021927306,0.6262081006044204,0.6492556427433388,0.6385317078175808],
# [0.3175345474532307,0.3165611073681377,0.32504497001169347,0.32475968907170794],
# [0.08847022702347788,0.10842147781298711,0.08315938986152124,0.10629568482423173],
# [0.12246779759518493,0.12835065120311132,0.1256905435295501,0.1288513564860433],
#           ]

# G = [[0.9393931755681177,0.10476822854025707,0.9880958192981563,0.10840506874369574],
# [1.322107801915166,0.15095670290805133,1.3597015658584362,0.16198989072768707],
# [2.375644052522172,0.33731352007549614,2.426288388700622,0.3378085116382122],
# [0.7401775326108267,0.2016126571694285,0.7448855988627049,0.20301655174255914],
# [1.2381227247577369,0.10063250795747439,1.2759004600444148,0.06722874808422617],
# [1.1904389968311175,0.12835065120311132,1.2164116205545503,0.1288513564860433],
# ]

# print(np.mean(G, axis =0))
# print(np.mean(M, axis =0))
# print(np.mean(S, axis =0))

####point 7
# idx = [[0, 9, 10, 1, 3, 5, 2],[8, 1, 0, 3, 10, 2, 5],

# ]
# M = [[1.413430510372179,1.438548424885435,1.448406511638475,1.473117050829723],
# [2.27625939725466,1.2625430169905036,2.343442232373083,1.300563125864634],
# [1.3051292029946822,1.1939422772022794,1.3451319980820282,1.221316221316303],
# [2.1587991714966086,1.4931224111207,2.205789241408784,1.4131622342305],
# [2.8783269590746614,2.1893350968841214,2.9281617517643728,2.2300267496041593],
# [3.4686214368225343,1.8310485094759539,3.5230057004583064,1.8746867599626897], 
# [2.189828944144463,1.7677882231294098,2.2329195902661954,1.8043975341297027],
# ]

# S = [ [0.153477454145573,0.16058862510644623,0.15153893346925698,0.1587910579375162],
# [0.353185182479742,0.08691208505190215,0.34032350027995356,0.09515473809978828],
#     [0.10397756772934003,0.06497670997406678,0.10023274785653402,0.06253050622346255],
#     [0.37865816900976645,0.2819128860279378,0.3972157193971927,0.3041373367689884],
#     ]

# G = [[1.200393656920548,0.18514987831525284,1.2507417168428934,0.18541190491065312],
# [1.2289510834626842,0.2412896217085611,1.2567249028542806,0.23987688392803108],
#     [0.8310485094759539,0.09346816206072395,0.8746867599626897,0.1012191459967214],
#     [1.74,0.2819,1.793344,0.32432454]
#     ]

# print(np.mean(G, axis =0))
# print(np.mean(M, axis =0))
# print(np.mean(S, axis =0))

##### point num = 8
# idx = [[9, 2, 4, 0, 5, 7, 1, 10],
# [4, 8, 0, 7, 2, 5, 9, 1],
#     [4, 9, 5, 1, 2, 10, 0, 3],
#     [7, 2, 10, 0, 3, 1, 5, 9],
#     [1, 8, 2, 9, 5, 10, 4, 0]


# ]

# M = [[1.2204011478921796,1.005171025725777,1.2719582573240472,1.0692499893782206],
# [2.3154087170547304,2.1659009833072007,2.3607957592134956,2.207843490546032],
# [2.5003416196610653,1.4246735518335203,2.5775120840395942,1.5041931614940875],
# [2.9418369507427453,1.7721832938273059,2.982418957590462,1.8129246504865],
# [1.8799057786009605,1.564519112765779,1.9266294375764401,1.5000695924889806],
# [2.4399841614801725,1.30809230001771,2.4763059870433752,1.3429864375241516],
# [2.7479892789898495,1.544078879933342,2.8096570846159408,1.6092676466722506],
# [2.244470223720133,1.8041217146555869,2.3075033354270225,1.850005288455978],]

# S = [ [0.16599112769910437,0.15512591221993766,0.16161569243812868,0.1622322471361162],
# [0.51747863647617,0.5069210242692372,0.5200421422986492,0.5108449532139342],
# [0.21418051828835322,0.2077737504444517,0.17208846181958276,0.19533971622246588],
# [0.2415029891443936,0.2216748403911326,0.24045034767631507,0.21914412938552075],
# [0.24007543351130914,0.23737709347801975,0.2392901527195456,0.24202552611876568],
# [0.3948247580534169,0.3344007555295195,0.40589020802272374,0.3441163725117796]

#  ]

# G = [[1.231204000455075,0.151357804313081,1.266522251707651,0.1519107377631723],
# [0.51747863647617,0.5069210242692372,0.5200421422986492,0.5108449532139342],
#     [1.075169316207692,0.17084177603488723,1.1143899786407638,0.17652246945295427],
#     [1.160843690982816,0.16821692119805337,1.2182566544574247,0.17724579891657255],
#    [1.30809230001771,0.1563372087577399,1.3429864375241516,0.16117800430714127],
#    [1.464519112765779,0.23737709347801975,1.5000695924889806,0.24202552611876568]
# ]

# print(np.mean(G, axis =0))
# print(np.mean(M, axis =0))
# print(np.mean(S, axis =0))


### point num 9
# idx = [[], [4, 10, 2, 7, 3, 1, 0, 5, 9],[],
# [5, 8, 7, 3, 10, 2, 0, 1, 4],
# ]

# M = [[1.40046413437572,1.3141093839810545,1.437228626918159,1.347903022942689],
# [1.8555639415341574,1.5624867275678218,1.8975210765000183,1.599749661054095],
# [3.069646811792058,2.832681558864931,3.1334843865986506,2.9049475392386284],
#     [1.6470832423207555,1.0953344676670727,1.6893090174936896,1.1327877025645337],
#     [2.9696226134712163,1.8348256844804376,3.011320870421448,1.87446562128198],
#     [2.9483150413019494,1.5836107698312583,2.99499113427337,1.658245629624554],
#     [1.9413210729207933,1.4315586780901522,2.016183782840946,1.520061837105615]
# ]

# S = [[0.09790060004080843,0.09427950284644268,0.10670578068368164,0.10349649486051947],
# [0.1663948100340576,0.08935298622147411,0.167892799359102,0.09204303488301771],
#     [0.15192562121247208,0.10054540826346708,0.16112027317354433,0.09959412721383003],
#     [0.11343392138572223,0.080206892009914,0.10632479017151411,0.08051041990580715],
#     [0.3368773147085972,0.33026124727152734,0.3382919540167307,0.33165618300646466],
#     [0.25695561655012733,0.2041817770087978,0.2777826771943896,0.23604165659042212],
#     ]
    
# G = [[1.0978366485642028,0.1848476879088761,1.1411818067810937,0.19065148713362925],
# [1.4168856758645976,0.2097343985877737,1.4434330359528909,0.2122242964250619],
#     [1.1814968496410926,0.15419836204789916,1.2227128206012784,0.15768276561452949]

# ]

# print(np.mean(G, axis =0))
# print(np.mean(M, axis =0))
# print(np.mean(S, axis =0))

G = [[1.678369435754027,1.7820046861494798],
[1.3210020901760768,1.4111328426409528],
    [1.7316522724590944,1.8419708209080998],
    [0.8189734523770527,0.8416117034350183],
    [1.7633389811562477,1.895255643091435]

]
A = [[3.277257345696467,3.5346208035095152],
[1.3745522586536272,1.4706445802766581],
    [2.1200574632154336,2.2096370087206827],
    [1.021900128810708,1.0637946449268885],
    [2.1970981969879455,2.3678536339301837]
    
]
P = [[3.7267712283262235,4.0278423894251025],
[1.7540683642256485,1.9670007129710835],
    [3.3213701492765786,3.5822944264070444],
    [2.0662981002518683,2.1338300006114637],
    [4.132897060956865,4.469397676991819]
  
    
]
print(np.mean(G, axis =0),np.std(G, axis =0))
print(np.mean(P ,axis =0),np.std(P, axis =0))
print(np.mean(A, axis =0),np.std(A, axis =0))