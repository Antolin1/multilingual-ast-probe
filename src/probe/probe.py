import torch.nn as nn
import torch


class TwoWordPSDProbe(Probe):
    """Computes squared L2 distance after projection by a matrix.

    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """
    def __init__(self, probe_rank, model_dim, device):
        print('Constructing TwoWordPSDProbe')
        super(TwoWordPSDProbe, self).__init__()
        self.probe_rank = probe_rank
        self.model_dim = model_dim
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


class OneWordPSDProbe(Probe):
    """Computes squared L2 norm of words after projection by a matrix."""
    def __init__(self, probe_rank, model_dim, device):
        print('Constructing OneWordPSDProbe')
        super(OneWordPSDProbe, self).__init__()
        self.probe_rank = probe_rank
        self.model_dim = model_dim
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)
        self.device = device

    def forward(self, batch):
        """ Computes all n depths after projection
        for each sentence in a batch.
        Computes (Bh_i)^T(Bh_i) for all i

        Args:
            batch: a batch of word representations of the shape
                    (batch_size, max_seq_len, representation_dim)

        Returns:
            A tensor of depths of shape (batch_size, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(transformed.view(batchlen* seqlen, 1, rank),
        transformed.view(batchlen* seqlen, rank, 1))
        norms = norms.view(batchlen, seqlen)
        return norms


class ParserProbe(Probe):
    def __init__(self, probe_rank, hidden_dim, number_labels_c, number_labels_u):
        print('Constructing ParserProbe')
        super(ParserProbe, self).__init__()
        self.probe_rank = probe_rank
        self.hidden_dim = hidden_dim
        self.number_vectors_c = number_labels_c
        self.number_vectors_u = number_labels_u
        self.proj = nn.Parameter(data=torch.zeros(self.hidden_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.vectors_c = nn.Parameter(data=torch.zeros(self.probe_rank, self.number_vectors_c))
        nn.init.uniform_(self.vectors_c, -0.05, 0.05)
        self.vectors_u = nn.Parameter(data=torch.zeros(self.probe_rank, self.number_vectors_u))
        nn.init.uniform_(self.vectors_u, -0.05, 0.05)

    def forward(self, batch):
        """
        Args:
            batch: a batch of word representations of the shape
                    (batch_size, max_seq_len, representation_dim)

        Returns:
            d_pred: (batch_size, max_seq_len - 1)
            scores_c: (batch_size, max_seq_len - 1, number classes_c)
            scores_u: (batch_size, max_seq_len, number classes_u)
        """
        transformed = torch.matmul(batch, self.proj)
        shift = transformed[:, 1:, :]
        diffs = shift - transformed[:, :-1, :]
        return (diffs**2).sum(dim=2), torch.matmul(diffs, self.vectors_c), torch.matmul(transformed, self.vectors_u)
