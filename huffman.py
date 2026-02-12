from collections import Counter
import heapq

class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_tree(freqs):
    heap = [Node(ch, freq) for ch, freq in freqs.items()]
    heapq.heapify(heap)
    if len(heap) == 1:  # Edge case: only one character
        only = heapq.heappop(heap)
        return Node(None, only.freq, only, None)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, parent)
    return heap[0]

def build_codes(root):
    codes = {}
    def dfs(node, path):
        if node.char is not None:
            codes[node.char] = path if path else "0"  # Handle single char
            return
        if node.left: dfs(node.left, path + "0")
        if node.right: dfs(node.right, path + "1")
    dfs(root, "")
    return codes

def bits_to_bytes(bitstr):
    padding = (8 - len(bitstr) % 8) % 8
    bitstr_padded = bitstr + ("0" * padding)
    b = int(bitstr_padded, 2).to_bytes(len(bitstr_padded) // 8, "big") if bitstr_padded else b""
    return b, padding

def bytes_to_bits(b, total_bits):
    if not b:
        return ""
    bitstr = bin(int.from_bytes(b, "big"))[2:].zfill(len(b) * 8)
    return bitstr[:total_bits]

def huffman_compress(text):
    if text == "":
        return b"", {}, 0, 0
    freqs = Counter(text)
    root = build_tree(freqs)
    code_map = build_codes(root)
    bitstr = "".join(code_map[ch] for ch in text)
    total_bits = len(bitstr)
    compressed_bytes, padding = bits_to_bytes(bitstr)
    return compressed_bytes, code_map, padding, total_bits

def huffman_decompress(compressed_bytes, code_map, padding, total_bits):
    if not compressed_bytes and not code_map:
        return ""
    rev_map = {v: k for k, v in code_map.items()}
    bitstr = bytes_to_bits(compressed_bytes, total_bits)
    out = []
    buff = ""
    for b in bitstr:
        buff += b
        if buff in rev_map:
            out.append(rev_map[buff])
            buff = ""
    return "".join(out)
