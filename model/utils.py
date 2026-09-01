def get_model_fn(model, train=False):

    def model_fn(x, sigma, germline=None, attention_mask=None):

        if train:
            model.train()
        else:
            model.eval()

        return model(x, sigma, germline=germline, attention_mask=attention_mask)

    return model_fn


def get_score_fn(model, train=False, sampling=False):

    if sampling:
        assert not train, "Must sample in eval mode"

    model_fn = get_model_fn(model, train=train)

    def score_fn(x, sigma, germline=None, attention_mask=None):

        sigma = sigma.reshape(-1)

        score = model_fn(
            x,
            sigma,
            germline=germline,
            attention_mask=attention_mask
        )

        if sampling:
            return score.exp()

        return score

    return score_fn