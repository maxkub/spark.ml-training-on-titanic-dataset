from pyspark import  SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

"""
File to be run in pyspark :
execfile('path_to_file/file.py')
"""

# load data in pandas dataFrame
data = pd.read_csv('/home/maxime/kaggle_data/titanic/train.csv')
print data.head()

# create a SQLContext with the sparkContext 'sc' in pyspark
sqlc = SQLContext(sc)

# create a pyspark dataFrame from the pandas df
df = sqlc.createDataFrame(data)

# preparing the dataset
df = df.drop('Cabin')
df = df.drop('Ticket')
df = df.drop('Name')
df = df.drop('PassengerId')

dfnoNaN = df.dropna()

avg_age = dfnoNaN.groupby().avg('Age').collect()[0][0]
print "avg(age) = ", avg_age
df = df.fillna(avg_age,subset=['Age'])


df = df.replace(['male','female'],['-1','1'],'Sex')
df = df.withColumn('Sex',df.Sex.cast('int'))

df = df.replace(['S','Q','C'],['-1','0','1'],'Embarked')
df = df.withColumn('Embarked',df.Embarked.cast('int'))
df.printSchema()

formula = RFormula(formula="Survived ~ Sex + Age + Pclass + Fare + SibSp + Parch",featuresCol="features",labelCol="label")
df = formula.fit(df).transform(df)
df.show()

print "count = " , df.count()


# Build the model
lr = LogisticRegression(maxIter=10, regParam=0.01)
model = lr.fit(df)

prediction = model.transform(df)

prediction.show(truncate=False)
