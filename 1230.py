import os
import re
import csv
import sys
import glob
import tqdm
import h5py
import pooch
import pydot
import pickle
import subprocess
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from Bio import SeqIO
import networkx as nx
from pathlib import Path
from natsort import natsorted
from scipy.linalg import expm
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from pyvis.network import Network
from scipy.linalg import eigh, inv
from networkx.drawing.nx_pydot import to_pydot


class Singlecell_analysis:
    def read_scflie(self, file):
        file_path = Path(file)
        if file_path.is_dir():
            print(f"Reading 10X format from: {file}")
            data = sc.read_10x_mtx(file, var_names='gene_symbols', cache=True)
        elif file_path.suffix == '.h5ad':
            print(f"Reading h5ad file: {file}")
            data = sc.read_h5ad(file)
        elif file_path.suffix == '.h5':
            print(f"Reading h5 file: {file}")
            with h5py.File(file, 'r') as f:
                data = sc.read_10x_h5(file)
        elif file_path.suffix == '.txt':
            print(f"Reading txt file: {file}")
            data = sc.read_text(file)
        return data

    def scanpy_analysis(self, adata, res):
        """
        Description: Single cell analyse, including normalisation, dimensionality reduction and cluster.

        Args:
            data (Anndata): Single cell sequencing h5ad file

        Returns:
            Anndata: Anndata dataframe
        """
        print("Start scanpy analyse")

        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.tsne(adata)
        sc.tl.leiden(adata, resolution = res, flavor="igraph", n_iterations=2, directed=False)
        sc.pl.umap(adata, color='leiden', title=obj_name, show=False,frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / f"{obj_name}_umap.pdf")
        sc.pl.tsne(adata, color='leiden', title=obj_name, show=False,frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / f"{obj_name}_tsne.pdf")
        sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon', key_added = "leiden", pts=True)
        markers = sc.get.rank_genes_groups_df(adata, group=None, key='leiden')
        markers['pts_delta'] = markers['pct_nz_group'] - markers['pct_nz_reference']
        markers['energy'] = markers.apply(lambda row: f"{int(row['pts_delta']*100000)}", axis=1)
        markers.to_csv(outdir / f"{obj_name}_all_markers.csv", index=False)
        markers['group'] = pd.to_numeric(markers['group'], errors='coerce')
        markers['energy'] = pd.to_numeric(markers['energy'], errors='coerce')
        top10_markers = markers.groupby('group', group_keys=False).apply(
            lambda group: group.nlargest(10, 'energy'), include_groups = True)
        top10_markers.to_csv(outdir / f"{obj_name}_top10_markers.csv", index=False)
        adata.write(outdir / f"{obj_name}_cluster.h5ad")
        print("End basic analyse")
        return adata, top10_markers

    def blastn(self, top10_markers, model_fasta, non_model_fasta):

        top10_output_dir = outdir / f"top10_genes_fasta"
        top10_output_dir.mkdir(parents=True, exist_ok=True)
        for cluster_id, group in top10_markers.groupby('group'):
            gene_list = group.iloc[:, 1].astype(str).tolist()
            output_file = top10_output_dir/f"cluster{cluster_id}_genes.fasta"
            with open(output_file, "w") as output_handle:
                for record in SeqIO.parse(non_model_fasta, "fasta"):
                    for gene in gene_list:
                        if gene == record.id:
                            SeqIO.write(record, output_handle, "fasta")

        blastn_output_dir = outdir / f"blastn_results"
        blastn_output_dir.mkdir(parents=True, exist_ok=True)
        for cluster_id, group in top10_markers.groupby('group'):
            cmd = f"blastn -query {top10_output_dir}/cluster{cluster_id}_genes.fasta -subject {model_fasta} -out {blastn_output_dir}/cluster{cluster_id}_genes.txt"
            cmd += f" -outfmt 6"
            print(cmd)
            subprocess.run(cmd, shell=True)

    def blastp(self, top10_markers, model_fasta, non_model_fasta):

        top10_output_dir = outdir / f"top10_genes_fasta"
        top10_output_dir.mkdir(parents=True, exist_ok=True)
        for cluster_id, group in top10_markers.groupby('group'):
            gene_list = group.iloc[:, 1].astype(str).tolist()
            output_file = top10_output_dir/f"cluster{cluster_id}_genes.fasta"
            with open(output_file, "w") as output_handle:
                for record in SeqIO.parse(non_model_fasta, "fasta"):
                    for gene in gene_list:
                        if gene == record.id:
                            SeqIO.write(record, output_handle, "fasta")

        blastp_output_dir = outdir / f"blastp_results"
        blastp_output_dir.mkdir(parents=True, exist_ok=True)
        for cluster_id, group in top10_markers.groupby('group'):
            cmd = f"blastp -query {top10_output_dir}/cluster{cluster_id}_genes.fasta -subject {model_fasta} -out {blastp_output_dir}/cluster{cluster_id}_genes.txt"
            cmd += f" -outfmt 6"
            print(cmd)
            subprocess.run(cmd, shell=True)
    

    def energy(self, dir, filename, cluster_id, top10_markers):
        file_path = os.path.join(dir, filename)
        print(file_path)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df = pd.read_csv(file_path, sep="\t", header=None)
            df = df.loc[df.groupby(0)[11].idxmax()]
            df = df[[0, 1]]
        else:
            df = pd.DataFrame()
            
        if df.empty:
            merged_df = pd.DataFrame()
        else:
            energy = top10_markers[top10_markers['group'] == str(cluster_id)][['names', 'energy']]
            merged_df = pd.merge(df, energy, left_on=(0), right_on='names', how='inner')
            merged_df = merged_df.iloc[:, [1, 3]]
            merged_df['Type'] = 'Gene'
            merged_df.columns = ['Gene_id', 'Energy', 'Type']
        return merged_df
    
    def cell_annotation(self,output_folder_path,adata):

        files = os.listdir(output_folder_path)
        txt_files = [f for f in files if f.endswith(".txt")]
        sorted_txt_files = sorted(txt_files, key = Q.natural_sort_key)
        # 打开汇总报告文件进行写入
        cluster_cell_type = {}
        for file_name_with_extension in sorted_txt_files:
            if file_name_with_extension.endswith(".txt"):
                # 构建每个 txt 文件的完整路径
                file_path = os.path.join(output_folder_path, file_name_with_extension)

                # 提取 cluster 编号
                cluster_id = file_name_with_extension.split('_')[0].replace('cluster', '')
                cluster_id = Path(cluster_id).stem
                #print(f'cluster{cluster_id}')
                cell_type = pd.read_csv(file_path)

                # 取第一行的 energy 值并计算其 25%
                threshold = cell_type.iloc[0]["energy"] * 0.5

                # 筛选出 energy 高于 threshold 的行
                filtered_rows = cell_type[cell_type["energy"] > threshold]
                if len(filtered_rows) == 1:
                    cluster_cell_type[f'cluster{cluster_id}'] = filtered_rows['name'].iloc[0]
                else:
                    node_id = filtered_rows['node_id'].to_list()
                    if organism == "Plant":
                        cypher = f"""
MATCH path = (a:Cell)-[b:PO_rel*1..2]-(c:Cell)
WHERE a.id IN {node_id}
AND c.id IN {node_id}
RETURN DISTINCT path
"""
                    elif organism == "Animal":
                        cypher = f"""
MATCH path =(a:Cell)-[b:develope_into|is_a*1..2]-(c:Cell)
WHERE a.id IN {node_id}
AND c.id IN {node_id}
RETURN DISTINCT path
"""
                    #print(cypher)
                    neo4j = KG.cypher_query_graph(cypher)
                    sub_g = KG.neo4j2networkx(neo4j)
                    if sub_g.number_of_edges() == 0:
                        cluster_cell_type[f'cluster{cluster_id}'] = filtered_rows['name'].iloc[0]
                    else:
                        # 计算所有节点的入度
                        in_degrees = sub_g.in_degree()
                        max_in_degree_node = max(in_degrees, key=lambda item: item[1])
                        max_in_degree = max_in_degree_node[1]
                        nodes_with_max_in_degree = [node for node, degree in in_degrees if degree == max_in_degree]
                        out_degrees = {node: sub_g.out_degree(node) for node in nodes_with_max_in_degree}
                        min_out_degree = min(out_degrees.values())
                        nodes_with_min_out_degree = [node for node, degree in out_degrees.items() if degree == min_out_degree]
                        if len(nodes_with_min_out_degree) > 1:
                            cluster_cell_type[f'cluster{cluster_id}'] = filtered_rows['name'].iloc[0]
                        else:
                            cluster_cell_type[f'cluster{cluster_id}'] = nodes_with_min_out_degree[0]

                
                
        cluster_cell_type = {key.replace('cluster', ''): value for key, value in cluster_cell_type.items()}
        adata.obs['cell_annotation'] = adata.obs['leiden'].map(cluster_cell_type)
        sc.pl.umap(adata, color='cell_annotation', title=obj_name, show=False)
        plt.tight_layout()
        plt.savefig(outdir / f"{obj_name}_cell_annotation_umap.pdf")
        sc.pl.tsne(adata, color='cell_annotation', title=obj_name, show=False)
        plt.tight_layout()
        plt.savefig(outdir / f"{obj_name}_cell_annotation_tsne.pdf")
        adata.write(outdir / f"{obj_name}_cell_annotation.h5ad")


class KGannotator:
    def __init__(self, url, auth):
        self.driver = GraphDatabase.driver(url, auth=auth)
                              
    def close(self):
        self.driver.close()    

    def cypher_query(self, cypher):
        with self.driver.session() as session:
            results = session.run(cypher)
            return [record for record in results]

    def cypher_query_graph(self, cypher):
        with self.driver.session() as session:
            results = session.run(cypher)
            return results.graph()
    
    def create_cypher_plant(self,resolution, source):
        cypher = ""
        if type(source) == dict:
            if resolution == "Cell":
                if "Gene" in source.keys():
                    cypher = f"""
UNWIND {source["Gene"]} AS geneName
MATCH path = (a:Gene)-[b:homologous_gene]-(c:Gene)-[d:marker_of]->(e:Cell)
WHERE a.id IN {source["Gene"]}
RETURN DISTINCT path
UNION
MATCH path = (a:Gene)-[b:marker_of]->(e:Cell)
WHERE a.id IN {source["Gene"]}
RETURN DISTINCT path
                    """
                        
            elif resolution == "Tissue":
                if "Gene" in source.keys():
                    cypher = f"""
UNWIND {source["Gene"]} AS geneName
MATCH path = (a:Gene)-[b:homologous_gene]-(c:Gene)-[d:marker_of]->(e:Cell)-[f:PO_rel{{relation: 'is_a'}}]-(g:Cell)
WHERE a.id IN {source["Gene"]}
RETURN DISTINCT path
UNION
MATCH path = (a:Gene)-[b:marker_of]->(e:Cell)-[f:PO_rel{{relation: 'is_a'}}]-(g:Cell)
WHERE a.id IN {source["Gene"]}
RETURN DISTINCT path
                    """
        else:
            print(f"{source} type error, current type is ",type(source))

        print(cypher)
        return cypher


    def create_cypher_animal(self, resolution, source):
        cypher = ""
        if type(source) == dict:
            if resolution == "Cell":
                if "Gene" in source.keys():
                    cypher = f"""
MATCH path = (a:Gene)-[b:marker_of]->(c:Cell)-[d:develope_into|is_a*1..2]->(e:Cell)
WHERE a.name IN {source["Gene"]} OR {source["Gene"]} IN SPLIT(a.synonym, '|')
RETURN DISTINCT path
                                """

            elif resolution == "Tissue":
                if "Gene" in source.keys():
                    cypher = f"""
MATCH path = (a:Gene)-[b:marker_of]->(c:Cell)-[d:develope_into|is_a*1..2]->(e:Cell)-[f:is_a]-(g:Tissue)
WHERE a.name IN {source["Gene"]} OR {source["Gene"]} IN SPLIT(a.synonym, '|')
RETURN DISTINCT path
                                """
                    
        else:  
            print(f"{source} type error, current type is ",type(source))
        
        print(cypher)
        return cypher


    def neo4j2networkx(self, neo4j_g):
        nodes = []
        edges = []
        for n in neo4j_g.relationships:
            x, y = n.nodes  

            if "id" in x and "id" in y:
                xid = x['id']
                yid = y['id']
                xname = x['name']  
                yname = y['name']
                xtype = x["type"]
                ytype = y["type"]
            elif "name" in x and "name" in y:
                xname = x['name']
                yname = y['name']
                xtype = x["type"]
                ytype = y["type"]
            
            if x is not None and y is not None:
                r = n.type
                rela_conf = n.get('relation_confidence', None)

                node_x = {
                    "name": xname if ":" not in xname else f'"{xname}"',
                    "id": xid if ":" not in str(xid) else f'"{xid}"',
                    "type": xtype if ":" not in xtype else f'"{xtype}"',
                    "energy": 0
                }
                node_y = {
                    "name": yname if ":" not in yname else f'"{yname}"',
                    "id": yid if ":" not in str(yid) else f'"{yid}"',
                    "type": ytype if ":" not in ytype else f'"{ytype}"',
                    "energy": 0
                }
                for node_dict in [node_x, node_y]: 
                    node_name = node_dict["name"]
                    node_attrs = {k: v for k, v in node_dict.items() if k!= "name"}
                    nodes.append((node_name, node_attrs))
                    
                # Ensure relation_confidence is a single value, taking max if it's a list
                if isinstance(rela_conf, list):
                    rela_conf = max(rela_conf) if rela_conf else 1
            
                edge_attrs = {
                    "type": r if ":" not in r else f'"{r}"',
                    "relation_confidence": 1 if rela_conf is None else (rela_conf if ":" not in str(rela_conf) else f'"{rela_conf}"'),
                    "color": 'black'
                }
                edges.append((xname if ":" not in xname else f'"{xname}"', yname if ":" not in yname else f'"{yname}"', edge_attrs))

        nxg = nx.MultiDiGraph()
        nxg.add_nodes_from(nodes)
        nxg.add_edges_from(edges)
        self.networkx_graph = nxg
        return nxg

    def visualize_network(self, nxg=None, save='nx.html', width = 1000, height = 800, **kwargs):
        if nxg is None:
            nxg = self.networkx_graph
        nt = Network(f'{height}px', f'{width}px', notebook=True, directed=True, **kwargs)
        nt.from_nx(nxg)
        nt.toggle_physics(False)
        nt.show(save)
        return nt
    
    def save(self, path, name, format="pkl", subgraph=None):
        graph = ""
        if subgraph == None:
            graph = to_pydot(self.networkx_graph)
        else:
            graph = to_pydot(subgraph)

        if format == "pkl":
            with open(f"{path}/{name}.pkl", 'wb') as f:
                pickle.dump(graph, f)
            print(f"Save in {path}/{name}.pkl")

        elif format == "graphml":
            nx.write_graphml(graph, f"{path}/{name}.graphml")
            print(f"Save in {path}/{name}.graphml")

        elif format == "pdf":
            graph.write_pdf(f"{path}/{name}.pdf")
            print(f"Save in {path}/{name}.pdf")

        elif format == 'svg':
            graph.write_svg(f"{path}/{name}.svg")
            print(f"Save in {path}/{name}.svg")

        elif format == 'csv':
            data_to_save = []
            for node in subgraph.nodes(data=True):
                node_name = node[0]
                node_attrs = node[1]
                node_id = node_attrs.get("id", "")
                node_type = node_attrs.get("type", "")
                energy = node_attrs.get("energy", 0)
                data_to_save.append([node_name, node_id, node_type, energy])

            with open(f"{path}/{name}", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Node Name', 'Node ID', 'Node Type', 'Energy'])
                writer.writerows(f"{path}/{name}.csv")
                print(f"Save in {path}/{name}.csv") 
        
        return f"{path}/{name}.{format}"

    def update_node_energy_from_df(self, merge_df, nx_graph):
        # 遍历 merge_df 数据框中的每一行
        for _, row in merge_df.iterrows():
            gene_id = row['Gene_id']
            energy = row['Energy']

            # 检查节点的 id 是否在 NetworkX 图中
            for node in nx_graph.nodes:
                if node == gene_id:  # 假设节点的名称或 id 与 Gene_id 匹配
                    nx_graph.nodes[node]['energy'] = energy  # 更新节点的 energy 属性
        return nx_graph


class Query:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.energy = {}

    def read_graph(self, filename):
        file_extension = filename.split('.')[-1].lower()
        if file_extension == 'graphml':
            self.graph = nx.read_graphml(filename)
        elif file_extension == 'pickle':
            with open(filename, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            print(f"不支持的文件格式: {file_extension}")
        return self.graph

    # 定义一个用于按数字自然顺序排序的函数
    def natural_sort_key(self,s):
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
    
    def read_txt_file(self, filename):
        data = []
        with open(filename, 'r') as file:
            for line in file:
                id_, energy, label = line.strip().split(',')
                data.append((id_, int(energy), label))
        return data
    
    def find_folders(self, folder_path):
        # 定义匹配模式
        folders = glob.glob(folder_path)
        file_path = []
        file_list = []

        # 遍历每个符合条件的文件夹
        for folder_path in folders:
            print(f"Start extracting Cell Type: {folder_path}")

            # 获取所有文件并进行排序
            files = os.listdir(folder_path)
            txt_files = [f for f in files if f.endswith(".txt")]
            sorted_txt_files = sorted(txt_files, key = Q.natural_sort_key)
            # 遍历文件夹中的所有 .txt 文件
            for file_name_with_extension in sorted_txt_files:
                # 只处理 .txt 文件
                if file_name_with_extension.endswith(".txt"):
                    # 构建文件的完整路径
                    path = os.path.join(folder_path, file_name_with_extension)
                    file_path.append(path)
                    file_list.append(file_name_with_extension)
        return file_path,file_list
    
    def create_output_folders(self, folder_path, folder_name):
        parent_folder_path = os.path.dirname(folder_path)
        output_folder_path = os.path.join(parent_folder_path, folder_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        return output_folder_path
    
    def extract_source(self, data):
        result_dict = {}
        result_energy_dict = {}

        for element in data:
            id_value, energy_value, label = element
            # 将id添加到对应label的列表中，如果label对应的列表不存在，则创建一个空列表
            result_dict.setdefault(label, []).extend(list(set([id_value])))
            
            if id_value not in result_energy_dict:
                result_energy_dict[id_value] = energy_value

        self.energy = result_energy_dict
        return result_dict,result_energy_dict
    
    def build_influence_matrix(self,sub_g=None):
        genes = []
        cells = []
        if sub_g != None:
            sg = sub_g
        else:
            sg = self.graph

        for node in sg.nodes(data=True):
            node_name = node[0]
            node_attrs = node[1]
            node_type = node_attrs.get("type", "")
            if node_type == "Gene":
                genes.append(node_name)
            elif node_type == "Cell":
                cells.append(node_name)

        n_genes = len(genes)
        n_cells = len(cells)

        infmat = np.zeros((n_genes + n_cells, n_genes + n_cells))
        for cell in cells:
            #connected_genes = list(sg.neighbors(cell)) #neighbors方法只检查出边
            connected_genes = list(sg.predecessors(cell))
            connection_count = len(connected_genes)
            for gene in connected_genes:
                gene_index = genes.index(gene)
                cell_index = n_genes + cells.index(cell)
                gene_index = genes.index(gene)
                cell_index = n_genes + cells.index(cell)
                #print(f"Gene: {gene}, Cell: {cell}, gene_index: {gene_index}, cell_index: {cell_index}")
                infmat[gene_index][cell_index] = connection_count
                #print(f"{infmat[gene_index][cell_index]}={connection_count}")
                infmat[cell_index][gene_index] = connection_count
                #print(f"{infmat[cell_index][gene_index]}={connection_count}")
        return genes,cells,infmat
    
    def set_energy(self, sub_g=None, energy=None):
        node2energy = {}
        if energy != None:
            eng = energy
        else:
            eng = self.energy

        if sub_g != None:
            g = sub_g
        else:
            g = self.graph

        for node in g.nodes(data=True):
            node_name = node[0]
            
            if node_name in eng:
                node2energy[node_name] = eng[node_name]
                node[1]['energy'] = eng[node_name]
            else:
                node2energy[node_name] = 0
    
        return node2energy

    def build_similarity_matrix_und(self, sub_g:None, energy:None):
        # 获取影响力矩阵
        genes,cells,infmat = self.build_influence_matrix(sub_g)
        # 设置节点初始能量并更新子图节点的能量属性
        node2energy = self.set_energy(sub_g,energy)
        h = np.array([node2energy[list(node2energy.keys())[i]] for i in range(len(node2energy))])
        indices = np.array([i for i in range(len(node2energy))])
        m = np.shape(infmat)[0]
        n = np.shape(h)[0]
        M = infmat[np.ix_(indices, indices)]
        M = np.minimum(M, M.transpose())
        sim_matrix = np.empty_like(M)
        for i in range(n):
            for j in range(i, n):
                sim_matrix[i, j] = max(h[i], h[j]) * M[i, j]
                sim_matrix[j, i] = sim_matrix[i, j]
        return genes,cells,sim_matrix
    
    def build_similarity_matrix_d(self, sub_g=None, energy=None):
        genes,cells,infmat = self.build_influence_matrix(sub_g)
        #print(infmat)
        #print(infmat.shape)
        # 设置节点初始能量并更新子图节点的能量属性
        node2energy = self.set_energy(sub_g, energy)
        #print(node2energy)
        h = np.array([node2energy[node_name] for node_name in node2energy])
        #print("h:",h)
        indices = np.array(list(range(len(node2energy))))
        #print("i:",indices)
        M = infmat[np.ix_(indices, indices)]
        #print("m:",M)
        # 有向图的相似性矩阵计算逻辑
        sim_matrix = M * h[:, np.newaxis]
        return genes,cells,infmat,node2energy,sim_matrix

    def expm_eig(self, A):
        D, V = eigh(A)
        return np.dot(np.exp(D) * V, inv(V))


    def hotnet2_diffusion(self, sub_g=None, beta=0.8, energy=None):
        """
        使用HotNet2算法对给定子图进行能量扩散。

        参数:
        sub_g: 要进行能量扩散的子图，若为None则使用整个图。
        beta: HotNet2算法中的扩散系数，取值范围通常在0到1之间，默认值为0.8。
        energy: 用于更新节点初始能量的字典，若为None则使用节点自身的初始能量设置。

        返回:
        经过能量扩散后的节点能量字典。
        """

        if sub_g is None:
            sub_g = self.graph

        # 计算图的邻接矩阵 W
        W = np.array(nx.adjacency_matrix(sub_g).todense())

        # 计算度矩阵 D
        degree_matrix = np.diag(np.sum(W, axis=0))  # 每列的和即为节点的度
        L = degree_matrix - W  # 拉普拉斯矩阵

        # 计算热核扩散矩阵
        heat_kernel_matrix = Q.expm_eig(-beta * L)

        # 假设node_energies是一个包含每个节点初始能量的数组
        #node_energies = np.array([sub_g.nodes[node].get('energy', 0) for node in sub_g.nodes])
        
        # 确保 node_energies 是 float 类型的 NumPy 数组
        node_energies = np.array([sub_g.nodes[node].get('energy', 0) for node in sub_g.nodes], dtype=float)

        # 如果 heat_kernel_matrix 不是 float 类型，也应将其转换为 float 类型
        if heat_kernel_matrix.dtype != float:
            heat_kernel_matrix = heat_kernel_matrix.astype(float)

        # 确认 heat_kernel_matrix 和 node_energies 的形状是否兼容
        if heat_kernel_matrix.shape[1] != node_energies.shape[0]:
            raise ValueError("The shapes of heat_kernel_matrix and node_energies are incompatible for dot product.")

        # 计算扩散后的能量
        diffused_energies = np.dot(heat_kernel_matrix, node_energies)

        # 计算扩散后的能量
        diffused_energies = np.dot(heat_kernel_matrix, node_energies)

        for idx, node in enumerate(sub_g.nodes):
            sub_g.nodes[node]['energy'] = diffused_energies[idx]
            #print(f"Node: {node}, Energy: {diffused_energies[idx]}")

        new_sub_g = sub_g
        return {node_name: {'energy': sub_g.nodes[node_name]['energy'], 
                             'name': sub_g.nodes[node_name]['id'], 
                             'type': sub_g.nodes[node_name]['type']
                               } for node_name in sub_g.nodes}, new_sub_g
        
    def energy_diffusion(self, sub_g):
        # 初始化扩散状态
        diffusion_state = {node: False for node in sub_g.nodes}  # 记录节点是否已经参与过扩散
        remaining_nodes = list(sub_g.nodes)  # 剩余尚未扩散的节点
        processed_nodes = []  # 存储已经处理过的节点

        active_nodes = [node for node in remaining_nodes if float(sub_g.nodes[node]['energy']) > 0]
        for node in active_nodes:
            energy = float(sub_g.nodes[node]['energy'])
            neighbors = sub_g.neighbors(node)
            total_confidence = sum(float(sub_g[u][v][0]['relation_confidence']) for u, v in sub_g.edges(node))
            for neighbor in neighbors:
                if diffusion_state[neighbor]:  # 如果邻居已经参与过扩散，跳过
                    continue
                #print(node, neighbor)
                relation_confidence = float(sub_g[node][neighbor][0]['relation_confidence'])
                energy_transfer = energy * (relation_confidence / total_confidence) if total_confidence > 0 else 0

                sub_g.nodes[neighbor]['energy'] += energy_transfer
                sub_g.nodes[node]['energy'] = 0  # 父节点能量为 0
                diffusion_state[node] = True
                processed_nodes.append(node)

        # 从剩余节点中移除已经处理过的节点
        remaining_nodes = [node for node in remaining_nodes if node not in processed_nodes]
        # 确保仍有能量的节点再次有机会扩散
        remaining_nodes = [node for node in remaining_nodes if sub_g.nodes[node]['energy'] > 0]

        new_sub_g = sub_g
        return {node_name: {'energy': sub_g.nodes[node_name]['energy'], 
                            'name': sub_g.nodes[node_name]['id'], 
                            'type': sub_g.nodes[node_name]['type']} for node_name in sub_g.nodes}, new_sub_g

    
    def find_top_2_connected_cells(self, sub_g):
        result_list = []
        genes = [node for node in sub_g.nodes() if sub_g.nodes[node].get('type') == 'Gene']
        for gene in genes:
            connected_cells = [(neighbor, sub_g.nodes[neighbor].get('energy'))
                            for neighbor in sub_g.neighbors(gene)
                            if sub_g.nodes[neighbor].get('type') == 'Cell']
            sorted_cells = sorted(connected_cells, key=lambda x: x[1], reverse=True)[:2]
            top_2_cells = [cell[0] for cell in sorted_cells]
            result_list.append((gene, top_2_cells))
        return result_list

    def save_result_to_txt(self, result_list, file_path):
        with open(file_path, 'w') as f:
            for gene, cells in result_list:
                if cells:  # 确保 cells 不为空
                    f.write(f"{gene}: {', '.join(cells)}\n")

    def write_report(self,output_folder_path,file_name):
        # 获取folder_path的上级目录
        parent_folder_path = os.path.dirname(output_folder_path)
        # 汇总报告文件路径
        summary_report_path = os.path.join(parent_folder_path, f"{file_name}.txt")
        # 打开汇总报告文件进行写入
        with open(summary_report_path, 'w') as summary_file:
            files = os.listdir(output_folder_path)
            txt_files = [f for f in files if f.endswith(".txt")]
            sorted_txt_files = sorted(txt_files, key = Q.natural_sort_key)
            # 遍历文件夹中的所有 txt 文件
            for file_name_with_extension in sorted_txt_files:
                if file_name_with_extension.endswith(".txt"):
                    # 构建每个 txt 文件的完整路径
                    file_path = os.path.join(output_folder_path, file_name_with_extension)

                    # 提取 cluster 编号
                    cluster_id = file_name_with_extension.split('_')[0].replace('cluster', '')
                    cluster_id = Path(cluster_id).stem
                    
                    cell_type = pd.read_csv(file_path)

                    # 取第一行的 energy 值并计算其 25%
                    threshold = cell_type.iloc[0]["energy"] * 0.33

                    # 筛选出 energy 高于 threshold 的行
                    filtered_rows = cell_type[cell_type["energy"] > threshold]

                    # 如果筛选结果少于 5 行，则取前 5 行
                    if len(filtered_rows) < 5:
                        filtered_rows = cell_type.head(5)

                    summary_file.write(f"Possible tissue or cell types for cluster {cluster_id}:\n")
                    # 使用 to_csv 方法创建一个 CSV 格式的字符串，不包含索引和列名，使用逗号分隔，值用双引号包围
                    csv_string = filtered_rows.to_csv(index=False, header=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL)
                    summary_file.write(csv_string)
                    summary_file.write('\n')  # 添加空行以区分不同cluster的结果

if __name__ == "__main__":

    res = 0.5
    resolution = "Cell"
    organism = "Animal"
    file = ""

    model_fasta = {
    "human": "./protein/result/human_GRCh38.p14_protein.fasta", 
    "mouse": "./protein/result/mouse_GRCm39_protein.fasta"
    }

    non_model_fasta = "./protein/result/bat_R.sinicus_protein.fasta"

#     url = "neo4j://localhost:7687"
#     auth = ("neo4j", "159753fzy")

    url = "neo4j://10.224.28.66:7687" 
    auth = ("neo4j", "bmVvNGpwYXNzd29yZA==")
    
    file_path = Path(file)
    obj_name = file_path.stem
    outdir = Path(f"./scanpy/")
    outdir.mkdir(parents=True, exist_ok=True)
    
    SC = Singlecell_analysis()
    data = SC.read_scflie(file)
    adata, top10_markers = SC.scanpy_analysis(data, res)
    
    KG = KGannotator(url, auth)
    Q = Query()
    all_merged_dfs = {}
    
    for species,model_fasta_path in model_fasta.items():
        os.makedirs(f"./scanpy/by_{species}/energy", exist_ok=True)
        outdir = Path(f"./scanpy/by_{species}")
        #SC.blastn(top10_markers, model_fasta_path, non_model_fasta)
        SC.blastp(top10_markers, model_fasta_path, non_model_fasta)
        #directory = f"{outdir}/blastn_results"
        directory = f"{outdir}/blastp_results"
        for cluster_id, filename in enumerate(natsorted(os.listdir(directory))):
            merged_df = SC.energy(directory, filename, cluster_id, top10_markers)
            if merged_df.empty:
                continue
            else:
                if len(model_fasta) > 1:
                    merged_clusters = {}
                    all_merged_dfs[f"cluster{cluster_id}_by_{species}"] = merged_df

                source = merged_df.groupby('Type')['Gene_id'].apply(list).to_dict()
                energy_int = dict(zip(merged_df['Gene_id'], merged_df['Energy']))
                if organism == "Animal":
                    cypher = KG.create_cypher_animal(resolution,source)
                elif organism == "Plant":
                    cypher = KG.create_cypher_plant(resolution,source)
                else:
                    print ("organism type undefined!")
                
                neo4j = KG.cypher_query_graph(cypher)
                sub_g = KG.neo4j2networkx(neo4j)
                file_name, _ = os.path.splitext(filename)
                nt = KG.visualize_network(sub_g,save=f'{outdir}/energy/cluster{cluster_id}.html')
                path = KG.save(f"{outdir}/energy",f"cluster{cluster_id}_sub_graph")
                sub_g = KG.update_node_energy_from_df(merged_df, sub_g)
                
                if len(sub_g.nodes()) == 0 or len(sub_g.edges()) == 0:
                    print(f"Skipping empty graph for {cluster_id}")
                    continue
                else:
                    #energy_distribution,new_sub_g = Q.hotnet2_diffusion(sub_g)
                    energy_distribution,new_sub_g = Q.energy_diffusion(sub_g)
                    output_file_path = f"{outdir}/energy/cluster{cluster_id}.txt"
                    with open(output_file_path, 'w') as file:
                        # 写入标题
                        file.write('"node_id","energy","name"\n')
                        # 将结果按能量从高到低排序并写入文件
                        for node, attrs in sorted(energy_distribution.items(), key=lambda item: item[1]['energy'], reverse=True):
                            if attrs["type"] in ["Cell", "Tissue"]:
                                file.write(f'{attrs["name"]},"{round(attrs["energy"], 2)}","{node}"\n')
                        print(f"Write in {output_file_path}")

                            
        Q.write_report(f"{outdir}/energy",f"cell_annotation_report")
        SC.cell_annotation(f"{outdir}/{obj_name}_cell_annotation_report.txt")

    if all_merged_dfs:
        temp_dict = {}
        # 遍历 all_merged_dfs 来分组相同前缀的数据框
        for cluster_name, df in all_merged_dfs.items():
            # 确定前缀，这里我们假设前缀是由'cluster'和数字组成，直到遇到非数字字符为止
            prefix = cluster_name.split('_')[0]

            # 如果前缀不在 temp_dict 中，则初始化一个空列表
            if prefix not in temp_dict:
                temp_dict[prefix] = []

            # 将数据框添加到相应前缀的列表中
            temp_dict[prefix].append(df)

        # 创建结果字典
        final_merged_dfs = {}

        # 合并具有相同前缀的数据框，并存储在 result_dict 中
        for prefix, dfs in temp_dict.items():
            # 使用 pd.concat 或者其他的合并方式，根据你的需求选择合适的参数
            merged_df = pd.concat(dfs, ignore_index=True)

            # 根据前缀生成新的键名
            new_key = f"{prefix}_multi"

            # 将合并后的数据框存入结果字典
            final_merged_dfs[new_key] = merged_df
            
        for cluster_id,merged_df in final_merged_dfs.items():
            match = re.match(r'cluster(\d+)_multi', cluster_id)
            cluster_id = int(match.group(1)) 

            os.makedirs(f"./scanpy/by_multi/energy", exist_ok=True)
            outdir = Path(f"./scanpy/by_multi")

            source = merged_df.groupby('Type')['Gene_id'].apply(list).to_dict()
            source['Gene'] = list(dict.fromkeys(source['Gene']))
            energy_int = dict(zip(merged_df['Gene_id'], merged_df['Energy']))
            if organism == "Animal":
                cypher = KG.create_cypher_animal(resolution,source)
            elif organism == "Plant":
                cypher = KG.create_cypher_plant(resolution,source)
            else:
                print ("organism type undefined!")
            neo4j = KG.cypher_query_graph(cypher)
            sub_g = KG.neo4j2networkx(neo4j)
            file_name, _ = os.path.splitext(filename)
            nt = KG.visualize_network(sub_g,save=f'{outdir}/energy/cluster{cluster_id}.html')
            path = KG.save(f"{outdir}/energy",f"cluster{cluster_id}_sub_graph")
            sub_g = KG.update_node_energy_from_df(merged_df, sub_g)

            if len(sub_g.nodes()) == 0 or len(sub_g.edges()) == 0:
                print(f"Skipping empty graph for cluster {cluster_id}")
                continue
            else:
                #energy_distribution,new_sub_g = Q.hotnet2_diffusion(sub_g)
                energy_distribution,new_sub_g = Q.energy_diffusion(sub_g)
                output_file_path = f"{outdir}/energy/cluster{cluster_id}.txt"
                with open(output_file_path, 'w') as file:
                    # 写入标题
                    file.write('"node_id","energy","name"\n')
                    # 将结果按能量从高到低排序并写入文件
                    for node, attrs in sorted(energy_distribution.items(), key=lambda item: item[1]['energy'], reverse=True):
                        if attrs["type"] in ["Cell", "Tissue"]:
                            file.write(f'{attrs["name"]},"{round(attrs["energy"], 2)}","{node}"\n')
                    print(f"Write in {output_file_path}")
                        
        Q.write_report(f"{outdir}/energy",f"cell_annotation_report")
        SC.cell_annotation(f"{outdir}/{obj_name}_cell_annotation_report.txt")

