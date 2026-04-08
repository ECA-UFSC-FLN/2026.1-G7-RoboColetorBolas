from Q1_Representacao import Graph

def floydWarshall(graph: Graph) -> list:
    quantVertices = graph.GetVerticesQuantity()
    D = [[float('inf')]*(quantVertices) for _ in range(quantVertices)]      #matriz com quantidade de vertices pela quantidade de vertices
    for i in range(quantVertices):
        for j in range(quantVertices):
            if i == j:                  #caso o indice seja igual (é a distancia do vertice para ele mesmo)
                D[i][j] = 0
            else:                       #caso seja diferente ele ve a distância para o outro vértice (representado pelo índice j+1)
                D[i][j] = graph.GetWeight(graph.GetLabel(i+1), graph.GetLabel(j+1))
    for i in range(quantVertices):      #após a matriz D estiver pronta, vai fazer o loop com a quantidade de vertices e cada loop vai colocar um vértice como fixo e ver a distância dele para outro
        D = matrizFW(D, i, quantVertices)
    return D, quantVertices             #retorna a matriz D

def matrizFW(D: list, k: int, quantVertices: int) -> list:
    D1 = [[float('inf')]*(quantVertices) for _ in range(quantVertices)]     #é criado uma segunda matriz que vai conter novas distancias minimas encontradas
    for i in range(quantVertices):                                          #é feita a verificação de todas as posições da matriz
        for j in range(quantVertices):
            if i == k or j == k or j==i:                                    #se for o vértice fixado é copiado da matriz anterior
                D1[i][j] = D[i][j]
            elif D[i][j] > D[i][k] + D[k][j]:                               #é verificado dij > dik + dkj 
                D1[i][j] = D[i][k] + D[k][j]                                #se for é implementado dij = dik + dkj (já que essa se torna a distancia minima atual)
            else:                                                           #se nao copia da matriz anterior
                D1[i][j] = D[i][j]
            
    return D1

if __name__ == "__main__":
    g = Graph()
    g.Read('GraphTest2.txt')
    D, quantVertices = floydWarshall(g)         #é retornada a matriz e a quantidade de vértices
    for i in range(quantVertices):              #intera-se sobre a quantidade de vertices para a impressão
        print(f'{i+1}:{",".join(map(str, D[i]))}')  #cada vértice é somado um (pois começa em 0 e os vértices começam em 1) e pego cada linha de D
