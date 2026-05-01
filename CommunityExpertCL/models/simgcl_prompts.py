"""
Instruction-prompt builder for the SimGCL baseline.

Adapted verbatim from ``LLM4GCL/common/prompts.py::get_instruction_prompts``;
the only change is internal dataset-name normalisation so the user's PyG
naming (``ogbn-arxiv``, ``ogbn-products``) maps to LLM4GCL's
(``arxiv``, ``products``).
"""

from textwrap import shorten

import numpy as np
import torch


_DATASET_ALIAS = {
    'ogbn-arxiv': 'arxiv',
    'ogbn-products': 'products',
}


def _normalize_dataset(dataset):
    return _DATASET_ALIAS.get(dataset, dataset)


def get_instruction_prompts(node_index, graph_data, text, label_index,
                            class_num, dataset, hop=(20, 20),
                            mode='neighbors', include_label=False,
                            max_node_text_len=256):
    """Build the (Context, Question, Answer) triple per requested node.

    Mirrors LLM4GCL's behaviour. ``graph_data`` must expose
    ``edge_index``, ``y`` and ``label_texts``; ``text`` is a list of
    raw texts indexed by node id (i.e. ``graph_data.raw_texts``).
    """
    dataset = _normalize_dataset(dataset)

    label_text_list = graph_data.label_texts
    label_text = label_text_list[: class_num]

    prefix_prompts = {
        'neighbors': (
            f"You are a good graph reasoner. Given a graph description "
            f"from {dataset} dataset, understand the structure and answer "
            f"the question.\n"),
        'ego': (
            f"You are a good graph reasoner. Given target node "
            f"information from {dataset} dataset, answer the question.\n"),
        'pure': (
            f"You are a good graph reasoner. Given a graph description "
            f"from {dataset} dataset, understand the structure and answer "
            f"the question.\n"),
    }

    question_prompts = {
        'cora': (f"Please predict which of the following sub-categories of "
                 f"AI does this paper belong to. Choose from the following "
                 f"{len(label_text)} categories: "
                 f"{', '.join([l.lower() for l in label_text])}"),
        'citeseer': (f"Please predict which of the following theme does this "
                     f"paper belong to. Choose from the following "
                     f"{len(label_text)} categories: "
                     f"{', '.join([l.lower() for l in label_text])}"),
        'wikics': (f"Please predict which branch of Computer Science this "
                   f"Wikipedia-based dataset belongs to. Choose from the "
                   f"following {len(label_text)} categories: "
                   f"{', '.join([l.lower() for l in label_text])}"),
        'photo': (f"Please predict which of the following categories does "
                  f"this photo item belong to. Choose from the following "
                  f"{len(label_text)} categories: "
                  f"{', '.join([l.lower() for l in label_text])}"),
        'products': (f"Please predict which of the following categories "
                     f"does this target item from Amazon belong to. Choose "
                     f"from the following {len(label_text)} categories: "
                     f"{', '.join([l.lower() for l in label_text])}"),
        'arxiv': (f"Please predict the most appropriate original arxiv "
                  f"identifier for the paper. Choose from the following "
                  f"{len(label_text)} categories: "
                  f"{', '.join([l.lower() for l in label_text])}."),
    }

    instructions = []
    for nid in node_index:
        if isinstance(nid, torch.Tensor):
            nid = int(nid.item())
        else:
            nid = int(nid)

        if dataset == 'products':
            node_id_token = f'Product id: {nid}\n'
        elif dataset == 'photo':
            node_id_token = f'Photo id: {nid}\n'
        elif dataset == 'wikics':
            node_id_token = f'Page id: {nid}\n'
        else:
            node_id_token = f'Paper id: {nid}\n'

        prefix = prefix_prompts[mode]
        context = prefix + '\n## Target node:\n' + node_id_token
        question = question_prompts[dataset] + (
            '\nDo not provide your reasoning.\n Answer:\n\n')
        answer = label_text[int(graph_data.y[nid].item())]

        if mode == 'neighbors':
            raw_text = shorten(text[nid], width=max_node_text_len, placeholder="...")
            context = f"{context}Text: {raw_text}\n"
            hop_neighbors = _get_subgraph(nid, graph_data.edge_index, hop)
            context += _get_structure_prompts(
                nid, label_index, label_text, graph_data, text,
                hop_neighbors, hop, include_label, dataset, mode,
                max_node_text_len)
        elif mode == 'ego':
            raw_text = shorten(text[nid], width=max_node_text_len, placeholder="...")
            context = f"{context}Text: {raw_text}\n"
        elif mode == 'pure':
            hop_neighbors = _get_subgraph(nid, graph_data.edge_index, hop)
            context += _get_structure_prompts(
                nid, label_index, label_text, graph_data, text,
                hop_neighbors, hop, include_label, dataset, mode,
                max_node_text_len)
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        instructions.append({
            "Context": context,
            "Question": question,
            "Answer": answer,
        })

    return instructions


def _get_subgraph(node_idx, edge_index, hop):
    current = torch.tensor([node_idx])
    hop_neighbors = []
    for _ in range(len(hop)):
        mask = (torch.isin(edge_index[0], current)
                | torch.isin(edge_index[1], current))
        new_nodes = torch.unique(torch.cat(
            (edge_index[0][mask], edge_index[1][mask])))
        diff = list(set(new_nodes.tolist()) - set(current.tolist()))
        hop_neighbors.append(diff)
        current = torch.unique(torch.cat((current, new_nodes)))
    return hop_neighbors


def _get_structure_prompts(node_index, label_idx, label_text, graph_data,
                           text, all_hop_neighbors, hop, include_label,
                           dataset, mode, max_node_text_len=256):
    out = ""
    if dataset == 'products':
        node_id_token = 'Product id: '
    elif dataset == 'photo':
        node_id_token = 'Photo id: '
    elif dataset == 'wikics':
        node_id_token = 'Page id: '
    else:
        node_id_token = 'Paper id: '

    for h in range(len(hop)):
        neighbors = np.unique(np.array(all_hop_neighbors[h]))
        np.random.shuffle(neighbors)
        if h == 0:
            neighbors = neighbors[:hop[0]]
        else:
            neighbors = neighbors[:hop[1]]

        if len(neighbors) == 0:
            continue

        if dataset == 'products':
            out += f"\nKnown neighbor products at hop {h + 1} (partial, may be incomplete):\n"
        elif dataset == 'photo':
            out += f"\nKnown neighbor photos at hop {h + 1} (partial, may be incomplete):\n"
        elif dataset == 'wikics':
            out += f"\nKnown neighbor pages at hop {h + 1} (partial, may be incomplete):\n"
        else:
            out += f"\nKnown neighbor papers at hop {h + 1} (partial, may be incomplete):\n"

        for nb in neighbors:
            nb_text = shorten(text[int(nb)], width=max_node_text_len, placeholder="...")
            if mode != 'pure':
                out += f"\n{node_id_token}{int(nb)}\nText: {nb_text}\n"
            else:
                out += f"\n{node_id_token}{int(nb)}"
            if include_label and int(nb) in label_idx:
                lbl = label_text[int(graph_data.y[int(nb)].item())]
                out += f"Label: {lbl}\n"
    return out
