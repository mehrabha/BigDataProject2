from project import spark

client = spark.Spark('./data/data.txt')
client.task1()
client.task2('')

#for elem in client.tfidf.collect():
    #print (' -', elem)
input('Press any key to continue...')