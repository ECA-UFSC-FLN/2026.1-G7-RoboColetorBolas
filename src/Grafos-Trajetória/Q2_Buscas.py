from Q1_Representacao import Graph
import queue

def BuscaLargura(graph: Graph, s: int):
    Cv = [] # Vertices visitados
    Dv = [] # Distancia em arestas para todos os vertices
    Av = [] # Antecessor do vertice no caminho definido a partir de (s)
    
    # Configurando todos os vértices
    for i in range(graph.GetVerticesQuantity()):
        Cv.append(False)
        Dv.append(float('inf'))
        Av.append(None)
    
    # Configurando vertice de origem
    Cv[s-1] = True
    Dv[s-1] = 0
    # Instanciando fila
    Q = queue.Queue()
    Q.put(s) # enfileira s
    
    while (Q.empty() == False):
        u = Q.get() 
        for neighbor in graph.GetNeighborhood(graph.GetLabel(u)):
            if Cv[graph.GetIndex(neighbor)-1] == False:
                Cv[graph.GetIndex(neighbor)-1] = True
                Dv[graph.GetIndex(neighbor)-1] = Dv[u-1] + 1
                Av[graph.GetIndex(neighbor)-1] = u
                Q.put(graph.GetIndex(neighbor))
    
    return (Dv,Av)

if __name__ == '__main__':
    g = Graph()
    g.Read('GraphTest2.txt')
    D, A = BuscaLargura(g, 1)
    print(f'D = {D}')
    print(f'A = {A}\n')
    
    # criando dicionario de niveis
    levels = {}
    for i in range(len(D)):
        level = D[i]
        vertex = i+1
        if level in levels:
            levels[level].append(vertex)
        else:
            levels[level] = [vertex]
    
    # Imprimindo a saída no formato do enunciado
    for level in levels:
        vertex_list = levels[level]
        vertex_list_str = ', '.join(str(v) for v in vertex_list)
        print(f"{level}: {vertex_list_str}")
            
        
    
     
    
    