class Graph():
    def __init__(self) -> None:
        self.adjList = {} # lista de adjacencias
        self.numEdges = 0
    
    def GetVerticesQuantity(self) -> int:
        return len(self.adjList)
    
    def GetEdgesQuantity(self) -> int:
        return self.numEdges
    
    def GetDegree(self, vertex: str) -> int:
        return len(self.adjList[vertex]['neighborhood'])
    
    def GetLabel(self, index: int) -> str:
        for v in self.adjList.items():
            if v[1]['index'] == index:
                return v[0]
    
    def GetIndex(self, label: str) -> int:
        return self.adjList[label]['index']
    
    def GetNeighborhood(self, vertex: str) -> list:
        return list(self.adjList[vertex]['neighborhood'].keys()) 
    
    def VerifyEdge(self, vertexU: str, vertexV: str) -> bool:
        if vertexV in self.adjList[vertexU]['neighborhood']:
            return True
        else:
            return False
    
    def GetWeight(self, vertexU: str, vertexV: str) -> float:
        try:
            weight = self.adjList[vertexU]['neighborhood'][vertexV]
        
        except:
            weight = float('inf')
            
        return weight
    
    def Read(self, file) -> None:
        with open(file, 'r') as f:
            lines = f.readlines()
            n = int(lines[0].split()[1]) # numero de vertices
            
            # Lendo vertices
            # Usado o numero de vertices para determinar o laço
            for i in range(1, n+1): 
                vertexLabel = lines[i].split()[1]
                
                # dicionario idenficado pelo rotulo do vertice, armazena sua vizinhança e seu index
                self.adjList.update({vertexLabel: {'neighborhood': {} , 'index': i}})
            
            # Lendo arestas
            # Lendo array de linhas do arquivo a partir de n+2
            # n+2 = numero de vertices + linhas com "*vertices" "*arestas"
            for line in lines[n+2:]: #
                self.numEdges += 1 # contando o numero de arestas
                
                vertexU, vertexV, weight = line.split()
                weight = float(weight)
                
                # dicionario de vizinhança, armazena o rotulo do vertice vizinho e o peso da aresta
                self.adjList[vertexU]['neighborhood'].update({vertexV: weight}) 
                self.adjList[vertexV]['neighborhood'].update({vertexU: weight})
                
if __name__ == '__main__':
    g = Graph()
    g.Read('GraphTest2.txt')
    print(g.adjList)
    print(g.GetVerticesQuantity())
    print(g.GetEdgesQuantity())
    print(g.GetDegree('a'))
    print(g.GetIndex('c'))
    print(g.GetLabel(4))
    print(g.GetNeighborhood('c'))
    print(g.VerifyEdge('a','c'))
    print(g.GetWeight('i','j'))