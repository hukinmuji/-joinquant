# -*- coding: utf-8 -*-
import jqdata
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime

class IndustrySW():
    def __init__(self):
        self.info ={'801010':{'name':'农林牧渔','pubdate':'2003-10-15'},
                    '801020':{'name':'采掘','pubdate': '2003-10-15'},
                    '801030':{'name':'化工','pubdate':'2003-10-15'},
                    '801040':{'name':'钢铁','pubdate':'2003-10-15'},
                    '801050':{'name':'有色金属','pubdate':'2003-10-15'},
                    '801080':{'name':'电子','pubdate':'2003-10-15'},
                    '801110':{'name':'家用电器','pubdate':'2003-10-15'},
                    '801120':{'name':'食品饮料','pubdate':'2003-10-15'},
                    '801130':{'name':'纺织服装','pubdate':'2003-10-15'},
                    '801140':{'name':'轻工制造','pubdate':'2003-10-15'},
                    '801150':{'name':'医药生物','pubdate':'2003-10-15'},
                    '801160':{'name':'公用事业','pubdate':'2003-10-15'},
                    '801170':{'name':'交通运输','pubdate':'2003-10-15'},
                    '801180':{'name':'房地产','pubdate':'2003-10-15'},
                    '801200':{'name':'商业贸易','pubdate':'2003-10-15'},
                    '801210':{'name':'休闲服务','pubdate':'2003-10-15'},
                    '801230':{'name':'综合','pubdate':'2003-10-15'},
                    '801710':{'name':'建筑材料','pubdate':'2014-2-21'},
                    '801720':{'name':'建筑装饰','pubdate':'2014-2-21'},
                    '801730':{'name':'电气设备','pubdate':'2014-2-21'},
                    '801740':{'name':'国防军工','pubdate':'2014-2-21'},
                    '801750':{'name':'计算机','pubdate':'2014-2-21'},
                    '801760':{'name':'传媒','pubdate':'2014-2-21'},
                    '801770':{'name':'通信','pubdate':'2014-2-21'},
                    '801780':{'name':'银行','pubdate':'2014-2-21'},
                    #'801790':{'name':'非银金融','pubdate':'2014-2-21'},
                    '801880':{'name':'汽车','pubdate':'2014-2-21'},
                    '801890':{'name':'机械设备','pubdate':'2014-2-21'}
                    }
    
    def industry_stocks_all(self,date = None,date_col = False): #返回全市场股票的行业信息,dataframe 格式
        df = pd.DataFrame()
        for code in self.info.keys():
            _df = pd.DataFrame(get_industry_stocks(code,date = date),columns =['code'])
            _df['industrycode'] = code
            _df['industryname'] = self.info[code]['name']
            if date_col:
                _df['date'] = date               
            df = pd.concat([df,_df],axis =0)
        return df

def get_valuebias(scandate,bygroup = True):
    print('计算价值偏离因子')
    #获得市值、账面价值、杠杆率数据
    q = query(valuation.code,valuation.market_cap*100000000,valuation.market_cap*100000000/valuation.pb_ratio,
              balance.total_liability/balance.equities_parent_company_owners)
    df = get_fundamentals(q,scandate)
    df.columns =['code','market_cap','bookvalue','leverage']
    df.set_index('code',inplace = True)
   
    #剔除负的bookvalue
    df = df[df['bookvalue'] > 0]
    
    #获得adjust_profit_ttm
    datelist = [str(dt) for dt in pd.date_range(start = '2000-01-01',end = scandate,freq ='Q').to_period('Q')][-6:]
    _df = pd.DataFrame()
    _q  = query(indicator.code,indicator.statDate,indicator.adjusted_profit).filter(indicator.pubDate <= scandate)
    for dt in datelist:
        _df = pd.concat([_df,get_fundamentals(_q,statDate = dt)],axis =0)

    _df.sort('statDate',inplace = True)
    profit = _df.groupby('code')['adjusted_profit'].apply(lambda x:x.iloc[-4:].sum() if x.count()>=4 else 0)
    
    #删除未公布四个季报
    profit = profit[profit<>0]
    df = pd.concat([df,profit],axis = 1,join ='inner')
    
    data = pd.DataFrame()
    data['market_cap']  = df['market_cap'].apply(lambda x:np.log(x))
    data['bookvalue' ]  = df['bookvalue'].apply(lambda x:np.log(x))
    data['leverage']    = df['leverage']
    data['profit_pos']  = df['adjusted_profit'].apply(lambda x: np.log(x) if x> 0 else 0)
    data['profit_']     = df['adjusted_profit'].apply(lambda x: np.log(x) if x> 0 else -np.log(-x))
    data['profit_neg']  = df['adjusted_profit'].apply(lambda x: np.log(-x) if x< 0 else 0)   #negative
    
    data = data.dropna()
    data = sm.add_constant(data,prepend = False)
    
    if bygroup:
        sw = IndustrySW()
        industries = sw.industry_stocks_all(date = scandate)
        industries.set_index('code',inplace = True)
        data = pd.concat([data,industries['industryname']],axis = 1,join ='inner').dropna()

    def _ols_epslon(data):
        y = data['market_cap']
        x = data[['const','bookvalue','leverage','profit_pos','profit_neg']]
        est = sm.OLS(y, x).fit()
        #print(est.summary())
        return  y - est.predict(x)
   
    if not bygroup:
        df = _ols_epslon(data)
        df = df.reset_index(drop = False)
        df.columns =['code','valuebias']
    else:
        df = data.groupby('industryname').apply(lambda x:_ols_epslon(x))
        df = df.reset_index(drop = False)
        df.columns =['industryname','code','valuebias']
    return df

# 初始化函数，设定基准等等
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    log.info('初始函数开始运行且全局只运行一次')
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    
    run_monthly(pe_ratio_median, 1,time='before_open')
    run_monthly(selection, 1,time='before_open') 
    run_monthly(rebalance, 1,time='open')

def pe_ratio_median(context):
    scandate = context.previous_date
    q = query(valuation.code, valuation.pe_ratio)
    df =get_fundamentals(q, scandate).dropna()
    df['ep_ratio']  = 1/df['pe_ratio']
    #pe中位数控制仓位
    context.pe_ratio_median = 1/df['ep_ratio'].quantile(0.5)
        
def selection(context):
    scandate = context.previous_date
    df = get_valuebias(scandate,bygroup = True)
    #剔除停牌及st
    stocklist = [stock for stock in df['code'] if not (get_current_data()[stock].paused or get_current_data()[stock].is_st)]
    df = df[df['code'].isin(stocklist)]
    df.set_index('code',inplace = True)

    #辅助过滤
    q = query(indicator.code,indicator.adjusted_profit,indicator.inc_revenue_year_on_year,indicator.inc_net_profit_to_shareholders_year_on_year,valuation.pb_ratio,valuation.pe_ratio)
    _df = get_fundamentals(q,scandate)
    _df.columns = ['code','adj_np','g_sale','g_np','pb','pe']
    _df.set_index('code',inplace = True)
    
    df = pd.concat([df,_df],axis = 1,join ='inner')
    df['q_valuebias'] = df.groupby('industryname')['valuebias'].apply(lambda x:pd.qcut(x,5,labels = False)+1)
    df['r_valuebias'] = df.groupby('industryname')['valuebias'].apply(lambda x:x.rank(ascending =True))
    df['q_g_sale'] = df.groupby('industryname')['g_sale'].apply(lambda x:pd.qcut(x,3,labels = False)+1)
    df['q_adj_np'] = df.groupby('industryname')['adj_np'].apply(lambda x:pd.qcut(x,2,labels = False)+1)
    df['q_g_np'] = df.groupby('industryname')['g_np'].apply(lambda x:pd.qcut(x,3,labels = False)+1)
    df['q_pb'] = df.groupby('industryname')['pb'].apply(lambda x:pd.qcut(1/x,3,labels = False)+1)
    df['q_pe'] = df.groupby('industryname')['pe'].apply(lambda x:pd.qcut(1/x,3,labels = False)+1)

    
    #参数进行了手动优化
    df = df[df['r_valuebias'] <= 5]    #选定28*5只股票作为备选池
    
    df = df[df['q_adj_np'] == 2]    #净利润绝对水平行业内前1/2
    
    df = df[df['g_sale'] >= 10]   #收入增速
    df = df[df['g_np'] >= 30]     #净利润增速
    df = df[(df['pe'] <= 22)&(df['pe'] > 0)]  
    df = df[df['pb'] <= 5]

    print(df.head())
    context.buylist = df.index.tolist()
    
def capital_allocation(context,buylist_tradable,method = 1):
    df = pd.DataFrame()
    scanDate = context.previous_date
    if method == 1: #pe_ratio
        q = query(valuation.code, valuation.pe_ratio).filter(valuation.code.in_(buylist_tradable))
        df =get_fundamentals(q, scanDate).dropna()
        df['ep_ratio'] = 1/df['pe_ratio']
        df['weight'] = df['ep_ratio']/df['ep_ratio'].sum()
 
    elif method == 2: #波动率加权
         import scipy.optimize as sco
         price = history(240, '1d', 'close', security_list = buylist_tradable)
         print(price)
         rets= np.log(price/ price.shift(1))
         def statistics(weights):

             weights= np.array(weights)
             pret= np.sum(rets.mean()* weights) * 240
             pvol= np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 240, weights)))

             return np.array([pret, pvol, pret/pvol])
         def min_func_variance(weights):
              return statistics(weights)[1] 
         cons= ({'type':'eq', 'fun':lambda x: np.sum(x)-1 })
         bnds= tuple((0, 1) for x in range(len(buylist_tradable)))

         opts= sco.minimize(min_func_variance, len(buylist_tradable) * [1./len(buylist_tradable)], bounds=bnds, method='SLSQP', constraints=cons)
         print('weight-->',opts['x'].round(3))
 

    elif method == 3:
         q = query(valuation.code, valuation.pe_ratio).filter(valuation.code.in_(buylist_tradable))
         df =get_fundamentals(q, scanDate).dropna()
         df['weight'] = 1.00/len(buylist_tradable)
    return df
       
## 开盘时运行函数
def rebalance(context):
    current_data = get_current_data()
    buylist_tradable = [stock for stock in context.buylist if not (current_data[stock].paused or current_data[stock].day_open >current_data[stock].high_limit*0.985)]
    print('buylist tradable ->',len(buylist_tradable))
    
    if  len(buylist_tradable) != 0:
        weights = capital_allocation(context,buylist_tradable)
        capital_allocation(context,buylist_tradable,method = 2)

    for stock in context.portfolio.positions.keys():
        if stock not in context.buylist:
            order_target(stock,0)
    
    print('pe median ->',context.pe_ratio_median)
    
    if  len(buylist_tradable) == 0:
        pass
    elif context.pe_ratio_median < 80:
        t_value = 0.95*context.portfolio.total_value
    elif context.pe_ratio_median > 80: 
        t_value = 0.20*context.portfolio.total_value
    
    for stock in buylist_tradable:
        weight = weights[weights['code']== stock]['weight']
        order_target_value(stock,weight*t_value)



