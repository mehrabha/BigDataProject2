from projectfiles import project2
import timeit
filesrc = './data/project2_egfr.txt'

client = project2.Project2(filesrc)
start = timeit.default_timer()
client.task1()
client.task2('gene_???_gene', 'gene_egfr+_gene')
stop = timeit.default_timer()
print('Time: ', stop - start)  
input('Press any key to continue...')