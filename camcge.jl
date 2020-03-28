#This model is a Cameroon General Equilibrium Model implemented using JULIA programming
#and its associated free packages such as ExcelReaders for reading in data, JuMP for
#the optimization of the problem and Ipopt which is a free solver

# Rewritten by John-Mary Matovu
#             Research Fellow
#             Research for Transformation and Development
#             Kampala, Uganda
#             www.rtdug.org

#This general equilibrium model is widely used as a blueprint
#for new model developments. It follows closely the style and type
#of model pioneered by Devis, De Melo and Robinson in the late 1970.
# and implented in GAMS by Condon, T, Dahl, H, and Devarajan, S
# Packages used (all free)

using XLSX
using JuMP
using Ipopt

# SETS

# Sectors

#  agsubsist, 'food crop
#  agexpind 'cash crops'
#  sylvicult  'forestry'
#  indalim   'food processing'
#  bienscons 'consumer goods'
#  biensint  'intermediate goods'
#  cimint    'construction materials'
#  bienscap  'capital goods'
#  construct  'construction'
#  services   'private services'
#  publiques  'public services'
SEC=[:agsubsist,:agexpind,:sylvicult,:indalim,:bienscons,:biensint,:cimint,:bienscap,:construct,:services,:publiques];
IT=[:agsubsist,:agexpind,:sylvicult,:indalim,:bienscons,:biensint,:cimint,:bienscap,:services];
ITN=[:construct,:publiques]

# Labour Catogories
lc=[:rural,:urbanunsk,:urbanskil];

# PARAMETERS
delta=Dict()   # 'Armington function share parameter                      (unity)'
ac   =Dict()   # 'Armington function shift parameter                      (unity)'
rhoc =Dict()   # 'Armington function exponent                             (unity)'
rhot =Dict()   # 'cet function exponent                                   (unity)'
at   =Dict()   # 'cet function shift parameter                            (unity)'
gamma=Dict()   # 'cet function share parameter                            (unity)'
eta  =Dict()   # 'export demand elasticity                                (unity)'
ad   =Dict()   # 'production function shift parameter                     (unity)'
cles =Dict()   # 'private consumption shares                              (unity)'
gles =Dict()   # 'government consumption shares                           (unity)'
depr =Dict()   # 'depreciation rates                                      (unity)'
dstr =Dict()   # 'ratio of inventory investment to gross output           (unity)'
kio  =Dict()   # 'shares of investment by sector of destination           (unity)'
tm0  =Dict()   # 'tariff rates                                            (unity)'
te   =Dict()   # 'export duty rates                                       (unity)'
itax =Dict()   # 'indirect tax rates                                      (unity)'
alphl=Dict()   # 'labor share parameter in production function            (unity)'

m0   =Dict()   # "volume of imports                             ('79-80 bill cfaf)"
e0   =Dict()   # "volume of exports                             ('79-80 bill cfaf)"
xd0  =Dict()   # "volume of domestic output by sector           ('79-80 bill cfaf)"
k0   =Dict()   # "volume of capital stocks by sector            ('79-80 bill cfaf)"
id0  =Dict()   # "volume of investment by sector of origin      ('79-80 bill cfaf)"
dst0 =Dict()   # "volume of inventory investment by sector      ('79-80 bill cfaf)"
int0 =Dict()   # "volume of intermediate input demands          ('79-80 bill cfaf)"
xxd0 =Dict()   # "volume of domestic sales by sector            ('79-80 bill cfaf)"
x0   =Dict()   # "volume of composite good supply               ('79-80 bill cfaf)"
pwe0 =Dict()   # "world market price of exports                            (unity)"

pwm0 =Dict()   # "world market price of imports                            (unity)"
pd0  =Dict()   # "domestic good price                                      (unity)"
pe0  =Dict()   # "domestic price of exports                                (unity)"
pm0  =Dict()   # "domestic price of imports                                (unity)"
pva0 =Dict()   # "value added price by sector                              (unity)"
qd   =Dict()   # "dummy variable for computing ad(i)                       (unity)"
xllb =Dict()   # "dummy variable (l matrix with no zeros)                  (unity)"
wa0  =Dict()   # "average wage rate by labor category ('79-80 mill cfaf pr worker)"

ld   =Dict()   # "employment                                        (1000 persons)"
ls0  =Dict()   # "labor supplies by category                        (1000 persons)";
it   =Dict()   # Import goods boolean indicator
in   =Dict()   # if not imported

cd0  =Dict()

# Base data
[wa0[l]=0.1 for l in lc];
wa0[:rural]      =  .11;
wa0[:urbanunsk] =  .15678;
wa0[:urbanskil] = 1.8657;

# ScalarS
er  = 0.21      #  real exchange rate                   (unity)"
gr0 = 179.0     #  government revenue        ('79-80 bill cfaf)"
gdtot0 = 135.03 #  government consumption    ('79-80 bill cfaf)"
cdtot0 = 947.98 #  private consumption       ('79-80 bill cfaf)"
fsav0=36.841    #  foreign saving         ('79-80 bill dollars)"
y0=0.0

# Data allocaated to dictiobaries
# Input Output Data
#dataio= readxl("camdata.xlsx", "iotable!B3:L13")
import XLSX

dataio= XLSX.readdata("C://Users/User/Documents/CGE-MAMS/CGE-JULIA/camdata.xlsx", "iotable", "B3:L13")
io=Dict()
for i=1:length(SEC), j=1:length(SEC)
    io[SEC[i],SEC[j]]=dataio[i,j];
end

# Capital Composition Matrix
datacap=XLSX.readdata("C://Users/User/Documents/CGE-MAMS/CGE-JULIA/camdata.xlsx", "imat!B3:L13")
imat=Dict();
for i=1:length(SEC), j=1:length(SEC)
    imat[SEC[i],SEC[j]]=datacap[i,j];
end

# Wage Proportionality factors
dataw=XLSX.readdata("C://Users/User/Documents/CGE-MAMS/CGE-JULIA/camdata.xlsx", "wagedist!B3:D13")
wdist=Dict()
for i=1:length(SEC), j=1:length(lc)
    wdist[SEC[i],lc[j]]=dataw[i,j];
end

# Employment by Sector and Wage Categories
datae=XLSX.readdata("C://Users/User/Documents/CGE-MAMS/CGE-JULIA/camdata.xlsx", "employment!B3:D13")
xle=Dict()
for i=1:length(SEC), j=1:length(lc)
    xle[SEC[i],lc[j]]=datae[i,j];
end

# Other Miscellaneous information
datazz=XLSX.readdata("C://Users/User/Documents/CGE-MAMS/CGE-JULIA/camdata.xlsx", "miscellaneous!B3:L19")
rowz=[:m0,:e0,:xd0,:k,:depr,:rhoc,:rhot,:eta,:pd0,:tm0,:itax,:cles,:gles,:kio,:dstr,:dst,:id];
zz=Dict()
for i=1:length(rowz), j=1:length(SEC)
    zz[rowz[i],SEC[j]]=datazz[i,j];
end

#Computation of parameters and coefficients for calibration
[depr[i]= zz[:depr,i] for i in SEC];
[rhoc[i]= 1/zz[:rhoc,i]- 1 for i in SEC];
[rhot[i]= 1/zz[:rhot,i] + 1 for i in SEC];
[eta[i] = zz[:eta,i] for i in SEC];
[tm0[i] = zz[:tm0,i] for i in SEC];
[te[i]  = 0 for i in SEC];
[itax[i]= zz[:itax,i] for i in SEC];
[cles[i] = zz[:cles,i] for i in SEC];
[gles[i]= zz[:gles,i] for i in SEC];
[kio[i] = zz[:kio,i] for i in SEC];
[dstr[i]= zz[:dstr,i] for i in SEC];
[xllb[i,l] = xle[i,l] + (1 - sign(xle[i,l])) for i in SEC, l in lc]
[m0[i] = zz[:m0,i] for i in SEC];
[e0[i] = zz[:e0,i] for i in SEC];
[xd0[i] = zz[:xd0,i] for i in SEC];
[k0[i]  = zz[:k,i] for i in SEC];
[pd0[i] = zz[:pd0,i] for i in SEC];
pm0     = pd0;
pe0     = pd0;
[pwm0[i]    = pm0[i]/((1 + tm0[i])*er) for i in SEC];
[pwe0[i]    = pe0[i]/((1 + te[i])*er) for i in SEC];
[pva0[i]    = pd0[i] - sum(io[j,i]*pd0[j] for j in SEC) - itax[i] for i in SEC];
[xxd0[i]    = xd0[i] - e0[i] for i in SEC];
[dst0[i]    = zz[:dst,i] for i in SEC];
[id0[i]     = zz[:id,i] for i in SEC];
[ls0[l]    = sum(xle[i,l] for i in SEC) for l in lc];
y0         = sum(pva0[i]*xd0[i] - depr[i]*k0[i] for i in SEC);
[cd0[i]    = cles[i]*cdtot0 for i in SEC];

[delta[i] = pm0[i]/pd0[i]*(m0[i]/xxd0[i])^(1+rhoc[i]) for i in IT];
[delta[i] = delta[i]/(1+delta[i]) for i in IT];
[x0[i]  = pd0[i]*xxd0[i] + pm0[i]*m0[i] for i in IT];
[x0[i]  = pd0[i]*xxd0[i]  for i in ITN];
[ac[i] =  x0[i]/(delta[i]*m0[i]^(-rhoc[i])+(1-delta[i])*xxd0[i]^(-rhoc[i]))^(-1/rhoc[i]) for i in IT];
[int0[i] = sum(io[i,j]*xd0[j] for j in SEC) for i in SEC];
[gamma[i]   = 1/(1 + pd0[i]/pe0[i]*(e0[i]/xxd0[i])^(rhot[i] - 1)) for i in IT];
[gamma[i]   = 0 for i in SEC if m0[i]==0];
[alphl[l,i] = (wdist[i,l]*wa0[l]*xle[i,l])/(pva0[i]*xd0[i]) for l in lc, i in SEC];

# get ad from output, ld from  profitmax, at from cet

[qd[i]  = (xllb[i,:rural]^alphl[:rural,i])*(xllb[i,:urbanunsk]^alphl[:urbanunsk,i])*(xllb[i,:urbanskil]^alphl[:urbanskil,i])*(k0[i]^(1 - sum(alphl[l,i] for l in lc))) for i in SEC];
[ad[i]  = xd0[i]/qd[i] for i in SEC];
[x0[i]  = pd0[i]*xxd0[i] + pm0[i]*m0[i] for i in IT];
[ac[i]  =  x0[i]/(delta[i]*m0[i]^(-rhoc[i])+(1-delta[i])*xxd0[i]^(-rhoc[i]))^(-1/rhoc[i]) for i in IT];

[ld[l] = sum((xd0[i]*pva0[i]*alphl[l,i]/(wdist[i,l]*wa0[l])) for i in SEC if wdist[i,l]>0 ) for l in lc ];

[at[i] = xd0[i]/(gamma[i]*e0[i]^rhot[i] + (1 - gamma[i])*xxd0[i]^rhot[i])^(1/rhot[i]) for i in IT];

# This is the time to lay out the model--this is where JuMP kicks in
function cammodel()
    #cgecam=Model(solver=IpoptSolver)
    cgecam=Model(with_optimizer(Ipopt.Optimizer))
        # Decision Variables
    @variables cgecam begin
        # prices block
        pd[i in SEC]>=1e-6, (start = pd0[i], base_name = "Domestic prices")
        pm[i in SEC]>=1e-6,  (start = pm0[i], base_name = "Domestic price of imports")
        pe[i in SEC]>=1e-6, (start = pe0[i], base_name = "domestic price of exports")
        pk[i in SEC]>=1e-6, (start = pd0[i], base_name = "rate of capital rent by sector")
        px[i in SEC]>=1e-6, (start = pd0[i], base_name = "average output price by sector")
        p[i in SEC]>=1e-6,  (start = pd0[i], base_name = "price of composite goods")
        pva[i in SEC]>=1e-6,(start = pva0[i], base_name = "value added price by sector")
        pwm[i in SEC]>=1e-6,(start = pwm0[i], base_name = "world market price of imports")
        pwe[i in SEC]>=1e-6,(start = pwe0[i], base_name = "world market price of exports")
        tm[i in SEC]>=1e-6, (start = tm0[i], base_name = "tariff rates")

        #production block
        x[i in SEC]>=1e-6,  (start = x0[i], base_name = "composite goods supply")
        xd[i in SEC]>=1e-6, (start = xd0[i], base_name = "domestic output by sector")
        xxd[i in SEC]>=1e-6,(start= xxd0[i], base_name = "domestic sales")
        e[i in SEC]>=1e-6,  (start=e0[i], base_name = "exports by sector")
        m[i in SEC]>=1e-6,  (start=m0[i], base_name = "imports ")

        # factors block
        k[i in SEC]>=1e-6, (start=k0[i], base_name=" capital stock by sector")
        wa[l in lc]>=1e-6, (start=wa0[l], base_name="average wage rate by labor category")
        ls[l in lc]>=1e-6, (start=ls0[l], base_name=" labor supply by labor category")
        labd[i in SEC,l in lc]>=1e-6, (start=xle[i,l], base_name= "employment by sector and labor category")

        # demand block
        int[i in SEC]>=1e-6, (start=int0[i], base_name="intermediates uses")
        cd[i in SEC]>=1e-6, (start=cd0[i], base_name=" demand for private consumption")
        gd[i in SEC]>=1e-6 #(start=g)# (lowerbound=1e-6, start=gd0)    #final demand for government consumption
        id[i in SEC]>=1e-6, (start=id0[i], base_name=" final demand for productive investment")
        dst[i in SEC]>=1e-6,(start=dst0[i], base_name= "inventory investment by sector")
        y>=1e-6, (start=y0, base_name = "private gdp")
        gr>=1e-6, (start=gr0, base_name = "government revenue")
        tariff>=1e-6, (start=76.548, base_name= "tariff revenue")
        indtax>=1e-6#, (start=102.45, base_name= "indirect tax revenue")
        duty>=1e-6     #export duty revenue
        gdtot>=1e-6#, (start= gdtot0, base_name= "total volume of government consumption")
        mps>=1e-6 #         (lowerbound=1e-6, start=0.1)    #marginal propensity to save
        hhsav>=1e-6 #       (lowerbound=1e-6, start=0.1)    #total household savings
        govsav>=1e-6 #     (lowerbound=1e-6, start=0.1)    #government savings
        deprecia>=1e-6 #     (lowerbound=1e-6, start=0.01)   #total depreciation expenditure
        savings>=1e-6#, (start= 280.98, base_name= "total savings")
        fsav>=1e-6, (start= fsav0, base_name="foreign savings")
        dk[i in SEC]>=1e-6# (lowerbound=1e-6, start=0.1)    #volume of investment by sector of destination
        # welfare indicator for objective function
        omega #        (lowerbound=1e-6, start=0.1)    #objective function variable
    end

    @NLconstraints cgecam begin
        # Prices
        pmdef[it in IT],        pm[it] == pwm[it]*er*(1 + tm[it]);
        pedef[it in IT],        pe[it]*(1 + te[it]) == pwe[it]*er;
        absorption1[i in IT],   p[i]*x[i]== pd[i]*xxd[i] + pm[i]*m[i];
        absorption2[itn in ITN],p[itn]*x[itn]== pd[itn]*xxd[itn];
        sales[i in SEC],        px[i]*xd[i] == pd[i]*xxd[i] + pe[i]*e[i];
        actp[i in SEC],         px[i]*(1-itax[i]) == pva[i] + sum(io[j,i]*p[j] for j in SEC);
        pkdef[i in SEC],        pk[i] == sum(p[j]*imat[j,i] for j in SEC);

        # output and factors of production block

        activity[i in SEC],     xd[i] == ad[i]*prod(labd[i,l]^alphl[l,i] for l in lc)*k[i]^(1 - sum(alphl[l,i] for l in lc));
        profitmax[i in SEC,l in lc], wa[l]*wdist[i,l]*labd[i,l] == xd[i]*pva[i]*alphl[l,i];

        lmequil[l in lc],       sum(labd[i,l] for i in SEC) == ls[l];

        cet[it in IT],          xd[it] == at[it]*(gamma[it]*e[it]^rhot[it] + ( 1 - gamma[it])
                                   * xxd[it]^rhot[it])^(1/rhot[it]);

        edemand[it in IT],      e[it]/e0[it]  == (pwe0[it]/pwe[it])^eta[it];

        esupply[it in IT],      e[it]/xxd[it] ==(pe[it]/pd[it]*(1 - gamma[it])/gamma[it])^(1/(rhot[it] - 1));

        armington[it in IT],    x[it] == ac[it]*(delta[it]*m[it]^(-rhoc[it]) + (1 - delta[it])
                                  * xxd[it]^(-rhoc[it]))^(-1/rhoc[it]);

        costmin[it in IT],      m[it]/xxd[it] ==(pd[it]/pm[it]*delta[it]/(1 - delta[it]))^(1/(1 + rhoc[it]));
        xxdsn[itr in ITN],      xxd[itr]== xd[itr];

        xsn[itr in ITN],        x[itr] == xxd[itr];

        # demand block
        inteq[j in SEC],        int[j] == sum(io[j,i]*xd[i] for i in SEC);

        dsteq[i in SEC],        dst[i] == dstr[i]*xd[i];

        cdeq[i in SEC],         p[i]*cd[i]  == cles[i]*(1 - mps)*y;

        gdp,                    y == sum(pva[i]*xd[i] for i in SEC) - deprecia;

        hhsaveq,                hhsav == mps*y;

        greq,                   gr == tariff + duty + indtax;

        gruse,                  gr == sum(p[i]*gd[i] for i in SEC) + govsav;

        gdeq[i in SEC],         gd[i] == gles[i]*gdtot;

        tariffdef,              tariff == sum(tm[it]*m[it]*pwm[it] for it in IT )*er;

        indtaxdef,              indtax == sum(itax[i]*px[i]*xd[i] for i in SEC);

        dutydef,                duty == sum(te[it]*e[it]*pe[it] for it in IT);

        depreq,                 deprecia == sum(depr[i]*pk[i]*k[i] for i in SEC);

        totsav,                 savings == hhsav + govsav + deprecia + fsav*er;

        prodinv[i in SEC],      pk[i]*dk[i] == kio[i]*savings - kio[i]*sum(dst[j]*p[j] for j in SEC);

        ieq[i in SEC],          id[i] == sum(imat[i,j]*dk[j] for j in SEC);

        caeq,                   sum(pwm[it]*m[it] for it in IT) == sum(pwe[it]*e[it] for it in IT) + fsav;

        #  market clearing
        equil[i in SEC],        x[i]== int[i] + cd[i] + gd[i] + id[i] + dst[i];

        obj,                    omega == prod(cd[i]^cles[i] for i in SEC if cles[i]>0.0);

        closurek[i in SEC],  k[i] == k0[i];
        closurep[i in SEC],  pwm[i] == pwm0[i];
        closurel[l in lc],   ls[l] == ls0[l] ;
        closuret[it in IT],  tm[it] == tm0[it];
        closuref,            fsav   == fsav0 ;
        closuremp,           mps    == 0.09305;
        closureg,            gdtot  == gdtot0;
        closurem[itn in ITN],m[itn]  == 0;
        closurelp,           labd[:publiques,:rural]== 0;
        closurela,           labd[:agsubsist,:urbanskil] == 0;
        closuree[itn in ITN],e[itn]  == 0;
    end
    @NLobjective(cgecam, Min, 1)
    #print(cgecam)
    #status=solve(cgecam)
    JuMP.optimize!(cgecam)
    pd=JuMP.value.(pd); pm=JuMP.value.(pm); pe=JuMP.value.(pe); pk=JuMP.value.(pk);
    px=JuMP.value.(px); p=JuMP.value.(p); pva=JuMP.value.(pva); pwm=JuMP.value.(pwm);
    pwe=JuMP.value.(pwe); tm=JuMP.value.(tm);

    x=JuMP.value.(x); xd=JuMP.value.(xd); xxd=JuMP.value.(xxd); e=JuMP.value.(e);
    m=JuMP.value.(m);

    k=JuMP.value.(k); wa=JuMP.value.(wa); ls=JuMP.value.(ls); labd=JuMP.value.(labd);

    int=JuMP.value.(int); cd=JuMP.value.(cd);
    gd=JuMP.value.(gd); id=JuMP.value.(id);dst=JuMP.value.(dst); y=JuMP.value.(y);
    gr=JuMP.value.(gr);

    tariff=JuMP.value.(tariff); indtax=JuMP.value.(indtax); duty=JuMP.value.(duty);
    gdtot=JuMP.value.(gdtot);mps=JuMP.value.(mps);hhsav=JuMP.value.(hhsav); govsav=JuMP.value.(govsav);

    deprecia=JuMP.value.(deprecia); savings=JuMP.value.(savings);
    fsav=JuMP.value.(fsav); dk=JuMP.value.(dk);

    return pd,pm,pe,pk,px,p,pva,pwm,pwe,tm,x,xd,xxd,e,m,k,wa,ls,labd,int,gd,id,dst,y,gr,tariff,indtax,duty,gdtot,mps,hhsav,govsav,deprecia,savings,fsav,dk;
end

# This structure will help in keeping data under one pocket. Use of Any is due
# to the fact that JuMP throws dictionaries that are not very usual.
struct Simulation
    pd::Any; pm::Any; pe::Any; pk::Any; px::Any; p::Any; pva::Any;
    pwm::Any; pwe::Any;tm::Any;x::Any;xd::Any;xxd::Any;e::Any;m::Any;
    k::Any;wa::Any;ls::Any;labd::Any;int::Any;gd::Any;id::Any;dst::Any;
    y::Any;gr::Any;tariff::Any;indtax::Any;duty::Any;gdtot::Any;mps::Any;
    hhsav::Any;govsav::Any;deprecia::Any;savings::Any;fsav::Any;dk::Any;
end

# Baseline Results
pd,pm,pe,pk,px,p,pva,pwm,pwe,tm,x,xd,xxd,e,m,k,wa,ls,labd,int,gd,id,dst,y,gr,tariff,indtax,duty,gdtot,mps,hhsav,govsav,deprecia,savings,fsav,dk = cammodel()
baseline=Simulation(pd,pm,pe,pk,px,p,pva,pwm,pwe,tm,x,xd,xxd,e,m,k,wa,ls,labd,int,gd,id,dst,y,gr,tariff,indtax,duty,gdtot,mps,hhsav,govsav,deprecia,savings,fsav,dk)

# Simulation 1
# Increase capital supply in agriculture sector by 10 percent
k0[:agsubsist]=1.10*k0[:agsubsist];
pd,pm,pe,pk,px,p,pva,pwm,pwe,tm,x,xd,xxd,e,m,k,wa,ls,labd,int,gd,id,dst,y,gr,tariff,indtax,duty,gdtot,mps,hhsav,govsav,deprecia,savings,fsav,dk = cammodel()
sim1=Simulation(pd,pm,pe,pk,px,p,pva,pwm,pwe,tm,x,xd,xxd,e,m,k,wa,ls,labd,int,gd,id,dst,y,gr,tariff,indtax,duty,gdtot,mps,hhsav,govsav,deprecia,savings,fsav,dk)
