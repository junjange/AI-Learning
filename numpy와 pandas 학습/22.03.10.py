import numpy as np
data1=[1,2,3,4,5]
data2=[1,2,3,3.5,4]
arr1=np.array(data1)
print(arr1,type(arr1))
print(arr1.shape)
arr2=np.array(data2)
arr3=np.array(data2)
print(arr1.dtype,arr2.dtype,arr3.dtype)
arr4=np.array([[1,2,3],[4,5,6,],[7,8,9],[10,11,12]])
print(arr4)
print(arr4.shape)
np.zeros((3,5))
np.ones(9)
np.ones((2,10))
np.arange(0,10,1)
np.arange(10)
np.arange(3,10)


arr1=np.array([[1,2,3],[4,5,6]])
arr1.shape
arr2=np.array([[10,11,12],[13,14,15]])
arr2.shape
print(arr1+arr2 )
print(arr1-arr2 )
print(arr1*arr2 )
print(arr1/arr2 )


#브로드 캐스트
arr3= np.array([10,11,12])
print(arr1.shape,arr3.shape)
print(arr1+arr3)
print(arr1*arr3)
print(10*arr1)
print(arr1**2)


#array 인덱싱
arr1=np.arange(10)
print(arr1[1])
print(arr1[3:9],arr1[:])
arr2=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arr2[2,:])
print(arr2[2,3])


#Array boolean
names=np.array(['a','b','c','a','c','d','e','f'])
print(names.shape)
data=np.random.randn(8,4)
print(data)
print(np.mean(data))
names_chanwoo_mask=(names =='a')#
print(names_chanwoo_mask)
print(data[names_chanwoo_mask,:])
print(data[names=='c',:])
print(data[(names=='d') | (names=='d'),:])
data[:,0]<0
data[data[:,0]<0,:]
#0번째 열의 값이 0보다 작은 행의 2,3번째 열 값
data[data[:,0]<0,2:4]
data[data[:,0]<0,2:4]=0


# numpy 함수
arr1=np.random.randn(5,3)
np.abs(arr1)
np.sqrt(arr1)
np.square(arr1)
np.exp(arr1)
np.log(arr1)
np.log10(arr1)
np.log2(arr1)
np.sign(arr1)
np.ceil(arr1)
np.floor(arr1)
np.isfinite(arr1)
np.cos(arr1)
arr2=np.random.randn(5,3)
np.multiply(arr1,arr2)
np.add(arr1,arr2)
np.subtract(arr1,arr2)
np.divide(arr1,arr2)
np.maximum(arr1,arr2)

#통계
np.sum(arr1)
np.sum(arr1,axis=1)
np.sum(arr1,axis=0)
np.mean(arr1)
np.mean(arr1, acis=0)
np.std(arr1)
np.min(arr1,axis=1)
np.sum(arr1,axis=1)
np.argmax(arr1,axis=0)
np.cumsum(arr1)
np.cumsum(arr1,axis=1)
np.cumpord(arr1)
np.sort(arr1)
np.sort(arr1)[::-1]
np.sort(arr1,axis=0)





# pandas

import numpy as np
import pandas as pd
obj =pd.Series([4,7,-5,3])
print(obj)
print(obj.values)
print(obj.index)
print(obj.dtype)
obj2 =pd.Series([4,7,-5,3] ,index=['a','b','c','d'])
sdata={'a':35000,'b':67000,'c':12000,'d':4000}
obj3=pd.Series(sdata)
print(obj3)
obj3.name ='Salary'
obj3.index.name='Names'
obj3.index=['A','B','C','D']


# data Frame

data={'name':['a','a','a','b','c'],'years' :[2013,2014,2015,2016,2015],'points':[1.5,1.7,3.6,2.4,2.9]}
df=pd.DataFrame(data)
print(df)
df.index
df.columns
df.values
df.index.name='Num'
df.columns.name='info'
print(df)
df2=pd.DataFrame(data,colimns=['year','name','points','penalty'],index=['one','two','three','four','five'])
print(df2)
print(df2.describe())
print(df2.info)
print(df2.info())



# DataFrame indexing
data={'names':['kilho','kilho','kilho','charies','charies'],'year':[2014,2015,2016,2015,2016],'points':[1.5,1.7,3.6,2.4,2.9]}
df=pd.DataFrame(data,columns=['year','names','points','penalty'],index=['one','two','three','four','five'])
print(df)
print(df['year'])
df.year
df[['year','points']]
df['penalty']=0.5
print(df)
df['penalty']=[0.1,0.2,0.3,0.4,0.5]
df['zeros']=np.arange(5)
print(df)
val=pd.Series([-1.2,-1.5,-1.7],index=['two','four','five'])
df['debt']=val
print(df)
df['net_points']=df['points']-df['penalty']
df['high_points']=df['net_points']>2.0
print(df)
del df['high_points']
df['net_points']
df['zeros']
print(df)
df.columns
df.index.name ='order'
df.columns.name='info'
print(df)

# DataFrame 행 선택 조작
print(df[0:3])
print(df['two':'four'])
print(df.loc['two'])
df.loc['two':'four']
df.loc['two':'four','points']
df.loc[:,'year']
df.loc[:,['year','names']]
df.loc['three':'five','year':'penalty']
df.loc['six',:]=[2013,'jun',4.0,0.1,2.1]
print(df)
df.ilco[3]
df.ilco[3:5,0:2]
df.iloc[[0,1,3],[1,2]]
df.ilco[:1,:4]
df.ilco[1,1]

# DataFrame boolean indexing
print(df)
print(df['year']>2014)
df.loc[df['year']>2014,:]
df.loc[df['names']== 'kilho',['names','points']]
print(df.loc[(df['points']>2) & (df['points']<3),:])
df.loc[df['points']>3,'penalty']=0
print(df)


# Data
df=pd.DataFrame(np.random.randn(6,4))
print(df)
df.columns=['A','B','C','D']
df.index=pd.date_range('20160701',periods=6)
print(df)
df['F']=[1.0,np.nan,3.5,6.1,np.nan,7.0]
print(df)
print(df.dropna(how='any'))
print(df.fillna(value=0.5))
df.isnull()
df.loc[df.isnull()['F'],:]
pd.to_datetime('20160701')
print(df.drop([pd.to_datetime('20160701')]))
df.drop([pd.to_datetime('20160702'),pd.to_datetime('20160704')])
df.drop('F',axis=1)
df.drop(['B','D'],axis=1)


# Data 분석용 함수들
data=[[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]]
df=pd.DataFrame(data, columns=["one","two"], index=["a","b","c","d"])
print(df)
df.sum(axis=0)
df.sum(axis=1)
df.sum(axis=1,skipna=False)
df['one'].sum()
df.loc['b'].sum()
df2=pd.DataFrame(np.random.randn(6,4),columns=["A","B","C","D"],index=pd.date_range("20160701",periods=6))
print(df2)
df2['A'].corr(df2['B'])
df2['A'].cov(df2['C'])
dates=df2.index
random_dates=np.random.permutation(dates)
df2=df2.reindex(index=random_dates,columns=["D","C","B","A"])
print(df2)
df2.sort_index(axis=0)
df2.sort_index(axis=1)
df2.sort_values(by='D')
df2['E']=np.random.randint(0,6,size=6)
df2['F']=["alpha","beta","gamma","gamma","alpha","gamma"]
print(df2)
df2.sort_values(by=['E','F'])
df2['F'].unique()
df2['F'].value_counts()
df2['F'].isin(["alpha","beta"])
df2.loc[df2['F'].isin(["alpha","beta"]),:]
df3=pd.DataFrame(np.random.randn(4,3),columns=["b","d","e"],index=['seoul','incheon','busan','daegu'])
print(df3)
func = lambda x:x.max()-x.min()
df3.apply(func, axis=0)
df3.to_csv('./data.csv')
df=pd.read_csv('./data.csv')
print(df)
 
