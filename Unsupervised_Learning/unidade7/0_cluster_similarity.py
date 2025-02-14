import numpy as np

def compute_single_linkage(cluster1, cluster2):
    """Calcula a menor distância entre dois clusters (ligação simples)."""
    return np.min([np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in cluster1 for p2 in cluster2])

def compute_complete_linkage(cluster1, cluster2):
    """Calcula a maior distância entre dois clusters (ligação completa)."""
    return np.max([np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in cluster1 for p2 in cluster2])

def compute_average_linkage(cluster1, cluster2):
    """Calcula a média das distâncias entre todos os pontos dos dois clusters."""
    distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in cluster1 for p2 in cluster2]
    return np.mean(distances)

def compute_centroid_linkage(cluster1, cluster2):
    """Calcula a distância entre os centroides dos dois clusters."""
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    return np.linalg.norm(centroid1 - centroid2)

# Definição dos clusters
cluster1 = np.array([[9.0,8.0],[6.0,4.0],[2.0,10.0],[3.0,6.0],[1.0,0.0]])
cluster2 = np.array([[7.0,4.0],[1.0,10.0],[6.0,10.0],[1.0,6.0],[7.0,1.0]])

# Cálculo das distâncias
print("Similaridade - Ligação Simples: ", compute_single_linkage(cluster1, cluster2))
print("Similaridade - Ligação Completa: ", compute_complete_linkage(cluster1, cluster2))
print("Similaridade - Ligação Média: ", compute_average_linkage(cluster1, cluster2))
print("Similaridade - Método do Centroide: ", compute_centroid_linkage(cluster1, cluster2))
