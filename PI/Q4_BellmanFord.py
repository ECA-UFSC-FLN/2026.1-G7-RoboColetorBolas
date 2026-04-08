from Q1_Representacao import Graph


def BellmanFord(graph: Graph, s: int) -> tuple():
    D = []  # Lista de pesos dos caminhos ( s , v )
    A = []  # Lista de antecessores

    # Inicialização
    for i in range(graph.GetVerticesQuantity()):
        D.append(float('inf'))
        A.append(None)

    D[s - 1] = float(0)

    # Relaxamento
    for _ in range(graph.GetVerticesQuantity() - 1):
        for vertex in range(1, graph.GetVerticesQuantity() + 1):  # Vertice
            for neighbor in graph.GetNeighborhood(graph.GetLabel(vertex)): # Vizinho
                # se a distancia até o vizinho do vertice atual do laço for maior do que a distancia
                # do vertice atual mais a aresta que leva a esse vizinho a condição é True
                if D[graph.GetIndex(neighbor) - 1] > D[vertex-1] + graph.GetWeight(graph.GetLabel(vertex), neighbor):
                    D[graph.GetIndex(neighbor) - 1] = D[vertex - 1] + graph.GetWeight(graph.GetLabel(vertex), neighbor)
                    A[graph.GetIndex(neighbor) - 1] = vertex

    # Verificação de ciclos negativos
    for vertex in range(1, graph.GetVerticesQuantity() + 1):
        for neighbor in graph.GetNeighborhood(graph.GetLabel(vertex)):
            if D[graph.GetIndex(neighbor) - 1] > D[vertex-1] + graph.GetWeight(graph.GetLabel(vertex), neighbor):
                return (False, None, None)

    return (True, D, A)


if __name__ == "__main__":
    g1 = Graph()
    g2 = Graph()
    g1.Read('GraphTest1.txt')  # grafo sem ciclos negativos
    g2.Read('GraphTest3.txt')  # grafo com ciclos negativos
    cycle1, D1, A1 = BellmanFord(g1, 6)
    cycle2, D2, A2 = BellmanFord(g2, 1)

    if cycle1:
        print('\nGrafo 1 não tem ciclo negativo\n')
        print(f'D1 = {D1}')
        print(f'A1 = {A1}\n')

        # Ajustando print no formato solicitado
        for i in range(1, g1.GetVerticesQuantity() + 1):
            path = []
            vertex = i
            while vertex is not None:
                path.insert(0, vertex)
                vertex = A1[vertex - 1]
            path_str = ','.join(str(v) for v in path)
            print(f"{i}: {path_str}; d={D1[i-1]}")

    else:
        print('\nGrafo 1 tem ciclo negativo\n')

    if cycle2:
        print("\nGrafo 2 não tem ciclo negativo\n")
        print(f'D2 = {D2}')
        print(f'A2 = {A2}\n')

        # Ajustando print no formato solicitado
        for i in range(1, g2.GetVerticesQuantity() + 1):
            path = []
            vertex = i
            while vertex is not None:
                path.insert(0, vertex)
                vertex = A2[vertex - 1]
            path_str = ','.join(str(v) for v in path)
            print(f"{i}: {path_str}; d={D2[i-1]}")

    else:
        print('\nGrafo 2 tem ciclo negativo\n')
