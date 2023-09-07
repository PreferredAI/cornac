import dgl
import torch


def construct_graph(data_set):
    """
    Generates graph given a cornac data set

    Parameters
    ----------
    data_set : cornac.data.dataset.Dataset
        The data set as provided by cornac
    """
    rating_row, rating_col, rating_values = data_set.uir_tuple

    rating_col = rating_col + data_set.total_users  # increment item id by num users

    rating_row, rating_col, rating_values = (
        torch.from_numpy(rating_row),
        torch.from_numpy(rating_col),
        torch.from_numpy(rating_values),
    )

    u, v = torch.cat(
        [rating_row, rating_col], dim=0
    ), torch.cat(
        [rating_col, rating_row], dim=0
    )

    # u, v = rating_row, rating_col

    # g = dgl.heterograph(
    #     {
    #         ("user", "rate", "item"): (rating_row, rating_col),
    #         # ("item", "item-user", "user"): (rating_col, rating_row),
    #     },
    #     num_nodes_dict={"user": data_set.num_users, "item": data_set.num_items}
    # )

    g = dgl.graph(
        (u, v),
        num_nodes=(data_set.total_users + data_set.total_items)
    )

    # g.edata["rates"] = {
    #     ("user", "user-item", "item"): rating_values,
    #     ("item", "item-user", "user"): rating_values,
    # }
    g.edata["rates"] = torch.cat((rating_values, rating_values), dim=0)

    # return dgl.to_bidirected(g)
    # return dgl.add_reverse_edges(g)
    return g
