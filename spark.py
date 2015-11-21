from pyspark import  SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.ml.feature import RFormula, StringIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt

"""
File to be run in pyspark :
execfile('path_to_file/file.py')
"""


def load_data(path):
    # load data in pandas dataFrame
    data = pd.read_csv(path)
    print data.head()

    # create a SQLContext with the sparkContext 'sc' in pyspark
    sqlc = SQLContext(sc)

    # create a pyspark dataFrame from the pandas df
    df = sqlc.createDataFrame(data)

    return df

def data_preparation(df, avg_age,feat_name="features",lab_name='label'):

    df = df.fillna(avg_age,subset=['Age'])

    """
    ## unnecessary when using Rformula
    df = df.replace(['male','female'],['-1','1'],'Sex')
    df = df.withColumn('Sex',df.Sex.cast('int'))

    df = df.replace(['S','Q','C'],['-1','0','1'],'Embarked')
    df = df.withColumn('Embarked',df.Embarked.cast('int'))
    df.printSchema()
    """

    # Rformula automatically formats categorical data (Sex and Embarked) into numerical data
    formula = RFormula(formula="Survived ~ Sex + Age + Pclass + Fare + SibSp + Parch",
        featuresCol=feat_name,
        labelCol=lab_name)

    df = formula.fit(df).transform(df)
    df.show(truncate=False)

    return df

def find_avg_age(df):
    df = df.drop('Cabin')
    df = df.drop('Ticket')
    df = df.drop('Name')
    df = df.drop('PassengerId')

    # filling missing value in Age with the average age
    dfnoNaN = df.dropna()
    avg_age = dfnoNaN.groupby().avg('Age').collect()[0][0]
    print "avg(age) = ", avg_age

    return avg_age

def buil_lrmodel(path):

    df = load_data(path)

    #-------------------- preparing the dataset -------------------------------------------

    avg_age = find_avg_age(df)
    df = data_preparation(df, avg_age)

    print "count = " , df.count()

    df = df.drop('Cabin')
    df = df.drop('Ticket')
    df = df.drop('Name')

    #------------------ Build a model ----------------------------------------------------
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    model = lr.fit(df)

    prediction = model.transform(df)
    prediction.show(truncate=False)

    evaluator = BinaryClassificationEvaluator()
    print "classification evaluation :" , evaluator.evaluate(prediction)


    #-------------- selecting models with cross validation -----------------------------------
    lr = LogisticRegression()
    grid = ParamGridBuilder().addGrid(lr.maxIter, [1,10,50,150,200,500,1000])\
                            .addGrid(lr.regParam, [0.01, 0.05, 0.1,]).build()
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
    cvModel = cv.fit(df)

    prediction = cvModel.transform(df)
    prediction.show(truncate=False)

    print "classification evaluation :" , evaluator.evaluate(prediction)


    return cvModel,avg_age

def build_decisionTree(path):

    df = load_data(path)
    avg_age=find_avg_age(df)
    df = data_preparation(df, avg_age)

    df = df.drop('Cabin')
    df = df.drop('Ticket')
    df = df.drop('Name')

    stringIndexer = StringIndexer(inputCol="Survived", outputCol="indexed")
    si_model = stringIndexer.fit(df)
    df = si_model.transform(df)
    df.show(truncate=False)

    dt = DecisionTreeClassifier(labelCol='indexed')
    grid = ParamGridBuilder().addGrid(dt.maxDepth, [1,5,10]).build()

    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=dt, estimatorParamMaps=grid, evaluator=evaluator)
    cvModel = cv.fit(df)

    prediction = cvModel.transform(df)
    prediction.show(truncate=False)

    print "classification evaluation :" , evaluator.evaluate(prediction)

    return cvModel,avg_age

def apply_onTest(model,avg_age,path):

    df = load_data(path)

    df = df.drop('Cabin')
    df = df.drop('Ticket')
    df = df.drop('Name')

    df = data_preparation(df, avg_age)

    print "count = " , df.count()

    prediction = model.transform(df)
    prediction.show(truncate=False)

    return prediction


if __name__ == "__main__":

    path = '/home/maxime/kaggle/spark.ml-training-on-titanic-dataset/'

    #model,mean_age = buil_lrmodel(path+'data/train.csv')
    model,mean_age = build_decisionTree(path+'data/train.csv')
    df = apply_onTest(model,mean_age,path+'data/test.csv')

    df = df.select('PassengerId','prediction')
    df = df.withColumnRenamed('prediction','Survived')
    df.show()

    df = df.toPandas()
    df['Survived']=df['Survived'].astype('int')
    df.to_csv(path+'results.csv',index=False)

    result = pd.read_csv(path+'data/genderclassmodel.csv')
    result.rename(columns={'Survived':'target','PassengerId':'Id'},inplace=True)
    df_comp = pd.concat([df, result], axis=1, join='inner')

    df_comp['diff']=df_comp.Survived - df_comp.target
    print df_comp.head()


    ax = df_comp.plot(kind='scatter', x='Id', y='target',color='r',s=100,label='target')
    df_comp.plot(kind='scatter',x='PassengerId',y='Survived',ax=ax,color='b',label='prediction')

    df_comp.plot(kind='scatter',x='PassengerId',y='diff')
    plt.show()
