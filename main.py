from projectfiles import project2
import timeit
filesrc = 'hdfs://localhost:9000/gene.txt'

client = project2.Project2(filesrc)
start = timeit.default_timer()
client.task1()
client.task2('gene_???_gene')
client.results()
stop = timeit.default_timer()
print('Time: ', stop - start)  
input('Press any key to continue...')