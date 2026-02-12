import zlib, heapq
from collections import defaultdict

# ---------------- Huffman ----------------
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    freq = defaultdict(int)
    for ch in text: freq[ch] += 1
    heap = [Node(ch, f) for ch, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l, r = heapq.heappop(heap), heapq.heappop(heap)
        merged = Node(None, l.freq + r.freq)
        merged.left, merged.right = l, r
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", codebook={}):
    if node is None: return
    if node.char is not None: codebook[node.char] = prefix
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_compress(text):
    if not text: return "", {}, None
    root = build_huffman_tree(text)
    codes = build_codes(root)
    encoded = "".join(codes[ch] for ch in text)
    return encoded, codes, root

def huffman_decompress(encoded, root):
    out = []
    node = root
    for bit in encoded:
        node = node.left if bit == "0" else node.right
        if node.char is not None:
            out.append(node.char)
            node = root
    return "".join(out)

# ---------------- Zlib ----------------
def zlib_compress(text): 
    return zlib.compress(text.encode("utf-8"))

def zlib_decompress(data): 
    return zlib.decompress(data).decode("utf-8")
