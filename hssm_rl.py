from torch.nn.modules.linear import Linear
from modules import *
from utils import *


class HierarchicalStateSpaceModel(nn.Module):
    def __init__(
        self,
        action_encoder,
        encoder,
        decoder,
        belief_size,
        state_size,
        num_layers,
        max_seg_len,
        max_seg_num,
        latent_n=10,
        use_min_length_boundary_mask=False,
        ddo=False,
        output_normal=True
    ):
        super(HierarchicalStateSpaceModel, self).__init__()
        ################
        # network size #
        ################
        # abstract level
        self.abs_belief_size = belief_size
        self.abs_state_size = belief_size
        self.abs_feat_size = belief_size

        # observation level
        self.obs_belief_size = belief_size
        self.obs_state_size = state_size
        self.obs_feat_size = belief_size

        # other size
        self.num_layers = num_layers
        self.feat_size = belief_size

        # sub-sequence information
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        # for concrete distribution
        self.mask_beta = 1.0

        #################################
        # observation encoder / decoder #
        #################################
        self.action_encoder = action_encoder
        self.enc_obs = encoder
        self.dec_obs = decoder
        self.combine_action_obs = nn.Linear(
            self.action_encoder.embedding_size + self.enc_obs.embedding_size,
            belief_size,
        )

        #####################
        # boundary detector #
        #####################
        self.prior_boundary = PriorBoundaryDetector(input_size=self.obs_feat_size)
        self.post_boundary = PostBoundaryDetector(
            input_size=self.feat_size, num_layers=self.num_layers, causal=True
        )

        #####################
        # feature extractor #
        #####################
        self.abs_feat = LinearLayer(
            input_size=self.abs_belief_size + self.abs_state_size,
            output_size=self.abs_feat_size,
            nonlinear=nn.Identity(),
        )
        self.obs_feat = LinearLayer(
            input_size=self.obs_belief_size + self.obs_state_size,
            output_size=self.obs_feat_size,
            nonlinear=nn.Identity(),
        )

        #########################
        # belief initialization #
        #########################
        self.init_abs_belief = nn.Identity()
        self.init_obs_belief = nn.Identity()

        #############################
        # belief update (recurrent) #
        #############################
        self.update_abs_belief = RecurrentLayer(
            input_size=self.abs_state_size, hidden_size=self.abs_belief_size
        )
        self.update_obs_belief = RecurrentLayer(
            input_size=self.obs_state_size + self.abs_feat_size,
            hidden_size=self.obs_belief_size,
        )

        #####################
        # posterior encoder #
        #####################
        self.abs_post_fwd = RecurrentLayer(
            input_size=self.feat_size, hidden_size=self.abs_belief_size
        )
        self.abs_post_bwd = RecurrentLayer(
            input_size=self.feat_size, hidden_size=self.abs_belief_size
        )
        self.obs_post_fwd = RecurrentLayer(
            input_size=self.feat_size, hidden_size=self.obs_belief_size
        )

        ####################
        # prior over state #
        ####################
        self.prior_abs_state = DiscreteLatentDistributionVQ(
            input_size=self.abs_belief_size, latent_n=latent_n
        )
        self.prior_obs_state = LatentDistribution(
            input_size=self.obs_belief_size, latent_size=self.obs_state_size
        )

        ########################
        # posterior over state #
        ########################
        self.post_abs_state = DiscreteLatentDistributionVQ(
            input_size=self.abs_belief_size + self.abs_belief_size,
            latent_n=latent_n,
            feat_size=self.abs_belief_size,
        )
        self.post_obs_state = LatentDistribution(
            input_size=self.obs_belief_size + self.abs_feat_size,
            latent_size=self.obs_state_size,
            output_normal=output_normal,
        )

        self.z_embedding = LinearLayer(
            input_size=latent_n, output_size=self.abs_state_size
        )

        self._use_min_length_boundary_mask = use_min_length_boundary_mask
        self.ddo = ddo
        self._output_normal = output_normal

    # sampler
    def boundary_sampler(self, log_alpha):
        # sample and return corresponding logit
        if self.training:
            log_sample_alpha = gumbel_sampling(log_alpha=log_alpha, temp=self.mask_beta)
        else:
            log_sample_alpha = log_alpha / self.mask_beta

        # probability
        log_sample_alpha = log_sample_alpha - torch.logsumexp(
            log_sample_alpha, dim=-1, keepdim=True
        )
        sample_prob = log_sample_alpha.exp()
        sample_data = torch.eye(2, dtype=log_alpha.dtype, device=log_alpha.device)[
            torch.max(sample_prob, dim=-1)[1]
        ]

        # sample with rounding and st-estimator
        sample_data = sample_data.detach() + (sample_prob - sample_prob.detach())

        # return sample data and logit
        return sample_data, log_sample_alpha

    # set prior boundary prob
    def regularize_prior_boundary(self, log_alpha_list, boundary_data_list):
        # only for training
        if not self.training:
            return log_alpha_list

        #################
        # sequence size #
        #################
        num_samples = boundary_data_list.size(0)
        seq_len = boundary_data_list.size(1)

        ###################
        # init seg static #
        ###################
        seg_num = log_alpha_list.new_zeros(num_samples, 1)
        seg_len = log_alpha_list.new_zeros(num_samples, 1)

        #######################
        # get min / max logit #
        #######################
        one_prob = 1 - 1e-3
        max_scale = np.log(one_prob / (1 - one_prob))

        near_read_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_read_data[:, 1] = -near_read_data[:, 1]
        near_copy_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_copy_data[:, 0] = -near_copy_data[:, 0]

        # for each step
        new_log_alpha_list = []
        for t in range(seq_len):
            ##########################
            # (0) get length / count #
            ##########################
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)
            seg_len = read_data * 1.0 + copy_data * (seg_len + 1.0)
            seg_num = read_data * (seg_num + 1.0) + copy_data * seg_num
            over_len = torch.ge(seg_len, self.max_seg_len).float().detach()
            over_num = torch.ge(seg_num, self.max_seg_num).float().detach()

            ############################
            # (1) regularize log_alpha #
            ############################
            # if read enough times (enough segments), stop
            new_log_alpha = (
                over_num * near_copy_data + (1.0 - over_num) * log_alpha_list[:, t]
            )

            # if length is too long (long segment), read
            new_log_alpha = over_len * near_read_data + (1.0 - over_len) * new_log_alpha

            ############
            # (2) save #
            ############
            new_log_alpha_list.append(new_log_alpha)

        # return
        return torch.stack(new_log_alpha_list, dim=1)

    # forward for reconstruction
    def forward(self, obs_data_list, action_list, seq_size, init_size):
        #############
        # data size #
        #############
        num_samples = action_list.size(0)
        full_seq_size = action_list.size(1)  # [B, S, C, H, W]

        #######################
        # observation encoder #
        #######################
        enc_obs_list = self.enc_obs(obs_data_list)  # [B, S, D]

        enc_action_list = self.action_encoder(action_list)

        # Shift sequence length dimension forward and 0 out first one
        shifted_enc_actions = torch.roll(enc_action_list, 1, 1)
        mask = torch.ones_like(shifted_enc_actions, device=shifted_enc_actions.device)
        mask[:, 0, :] = 0
        shifted_enc_actions = shifted_enc_actions * mask

        enc_combine_obs_action_list = self.combine_action_obs(
            torch.cat((enc_action_list, enc_obs_list), -1)
        )
        shifted_combined_action_list = self.combine_action_obs(
            torch.cat((shifted_enc_actions, enc_obs_list), -1)
        )

        ######################
        # boundary sampling ##
        ######################
        post_boundary_log_alpha_list = self.post_boundary(shifted_combined_action_list)
        boundary_data_list, post_boundary_sample_logit_list = self.boundary_sampler(
            post_boundary_log_alpha_list
        )
        boundary_data_list[:, : (init_size + 1), 0] = 1.0
        boundary_data_list[:, : (init_size + 1), 1] = 0.0

        if self._use_min_length_boundary_mask:
            mask = torch.ones_like(boundary_data_list)
            for batch_idx in range(boundary_data_list.shape[0]):
                reads = torch.where(boundary_data_list[batch_idx, :, 0] == 1)[0]
                prev_read = reads[0]
                for read in reads[1:]:
                    if read - prev_read <= 2:
                        mask[batch_idx][read] = 0
                    else:
                        prev_read = read

            boundary_data_list = boundary_data_list * mask
            boundary_data_list[:, :, 1] = 1 - boundary_data_list[:, :, 0]

        boundary_data_list[:, : (init_size + 1), 0] = 1.0
        boundary_data_list[:, : (init_size + 1), 1] = 0.0
        boundary_data_list[:, -init_size:, 0] = 1.0
        boundary_data_list[:, -init_size:, 1] = 0.0

        ######################
        # posterior encoding #
        ######################
        abs_post_fwd_list = []
        abs_post_bwd_list = []
        obs_post_fwd_list = []
        abs_post_fwd = action_list.new_zeros(num_samples, self.abs_belief_size).float()
        abs_post_bwd = action_list.new_zeros(num_samples, self.abs_belief_size).float()
        obs_post_fwd = action_list.new_zeros(num_samples, self.obs_belief_size).float()
        for fwd_t, bwd_t in zip(range(full_seq_size), reversed(range(full_seq_size))):
            # forward encoding
            fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)  # (B, 1)
            abs_post_fwd = self.abs_post_fwd(
                enc_combine_obs_action_list[:, fwd_t], abs_post_fwd
            )  # abs_post_fwd is psi for z
            obs_post_fwd = self.obs_post_fwd(
                enc_obs_list[:, fwd_t], fwd_copy_data * obs_post_fwd
            )  # obs_post_fwd is phi for s
            abs_post_fwd_list.append(abs_post_fwd)
            obs_post_fwd_list.append(obs_post_fwd)
            # backward encoding
            bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
            abs_post_bwd = self.abs_post_bwd(
                enc_combine_obs_action_list[:, bwd_t], abs_post_bwd
            )
            abs_post_bwd_list.append(abs_post_bwd)
        abs_post_bwd_list = abs_post_bwd_list[::-1]

        #############
        # init list #
        #############
        obs_rec_list = []
        prior_abs_state_list = []
        post_abs_state_list = []
        prior_obs_state_list = []
        post_obs_state_list = []
        prior_boundary_log_alpha_list = []
        selected_option = []
        onehot_z_list = []
        abs_state_list = []
        vq_loss_list = []

        #######################
        # init state / latent #
        #######################
        abs_belief = action_list.new_zeros(num_samples, self.abs_belief_size).float()
        abs_state = action_list.new_zeros(num_samples, self.abs_state_size).float()
        obs_belief = action_list.new_zeros(num_samples, self.obs_belief_size).float()
        obs_state = action_list.new_zeros(num_samples, self.obs_state_size).float()
        # this zero is ignored because first time step is always read
        p = torch.zeros(num_samples, self.post_abs_state.latent_n).to(abs_state.device)

        ######################
        # forward transition #
        ######################
        option = p
        for t in range(init_size, init_size + seq_size):
            #####################
            # (0) get mask data #
            #####################
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)

            #############################
            # (1) sample abstract state #
            #############################

            abs_belief = abs_post_fwd_list[t - 1] * 0

            vq_loss, z, perplexity, onehot_z, z_logit = self.post_abs_state(
                concat(abs_post_fwd_list[t - 1], abs_post_bwd_list[t])
            )
            abs_state = read_data * z + copy_data * abs_state

            abs_feat = self.abs_feat(
                concat(abs_belief, abs_state)
            )
            selected_state = np.argmax(
                onehot_z.detach().cpu().numpy(), axis=-1
            )  # size of batch
            onehot_z_list.append(onehot_z)

            ################################
            # (2) sample observation state #
            ################################
            obs_belief = read_data * self.init_obs_belief(
                abs_feat
            ) + copy_data * self.update_obs_belief(
                concat(obs_state, abs_feat), obs_belief
            )  # this is h
            obs_belief *= 0
            prior_obs_state = self.prior_obs_state(obs_belief)
            if self._output_normal:
              post_obs_state = self.post_obs_state(concat(enc_obs_list[:, t], abs_feat))
            else:
              # Use recurrent embedder
              post_obs_state = self.post_obs_state(concat(obs_post_fwd_list[t], abs_feat))

            if self._output_normal:
              if self.ddo:
                  obs_state = post_obs_state.mean
              else:
                  obs_state = post_obs_state.rsample()
            else:
              obs_state = post_obs_state
            obs_feat = self.obs_feat(concat(obs_belief, obs_state))

            ##########################
            # (3) decode observation #
            ##########################
            obs_rec_list.append(obs_feat)

            ##################
            # (4) mask prior #
            ##################
            prior_boundary_log_alpha = self.prior_boundary(obs_feat)

            ############
            # (5) save #
            ############
            prior_boundary_log_alpha_list.append(prior_boundary_log_alpha)
            # prior_abs_state_list.append(prior_abs_state)
            # post_abs_state_list.append(post_abs_state)
            prior_obs_state_list.append(prior_obs_state)
            post_obs_state_list.append(post_obs_state)
            selected_option.append(selected_state)
            abs_state_list.append(abs_state)
            vq_loss_list.append(vq_loss)

        # decode all together
        obs_rec_list = torch.stack(obs_rec_list, dim=1)

        obs_rec_list = self.dec_obs(obs_rec_list.view(num_samples * seq_size, -1))

        # (batch_size, sequence length, action size)
        obs_rec_list = obs_rec_list.view(num_samples, seq_size, -1)

        # stack results
        prior_boundary_log_alpha_list = torch.stack(
            prior_boundary_log_alpha_list, dim=1
        )

        # remove padding
        boundary_data_list = boundary_data_list[:, init_size : (init_size + seq_size)]
        post_boundary_log_alpha_list = post_boundary_log_alpha_list[
            :, (init_size + 1) : (init_size + 1 + seq_size)
        ]
        post_boundary_sample_logit_list = post_boundary_sample_logit_list[
            :, (init_size + 1) : (init_size + 1 + seq_size)
        ]

        # fix prior by constraints
        prior_boundary_log_alpha_list = self.regularize_prior_boundary(
            prior_boundary_log_alpha_list, boundary_data_list
        )

        # compute log-density
        prior_boundary_log_density = log_density_concrete(
            prior_boundary_log_alpha_list,
            post_boundary_sample_logit_list,
            self.mask_beta,
        )
        post_boundary_log_density = log_density_concrete(
            post_boundary_log_alpha_list,
            post_boundary_sample_logit_list,
            self.mask_beta,
        )

        # compute boundary probability
        prior_boundary_list = F.softmax(
            prior_boundary_log_alpha_list / self.mask_beta, -1
        )[..., 0]
        post_boundary_list = F.softmax(
            post_boundary_log_alpha_list / self.mask_beta, -1
        )[..., 0]
        prior_boundary_list = Bernoulli(probs=prior_boundary_list)
        post_boundary_list = Bernoulli(probs=post_boundary_list)
        boundary_data_list = boundary_data_list[..., 0].unsqueeze(-1)

        # process options
        selected_option = np.stack(selected_option).transpose((1, 0))  # size (B, S)
        onehot_z_list = torch.stack(onehot_z_list, axis=1)  # (B, S, Z)

        # process vq loss
        vq_loss_list = torch.stack(vq_loss_list)

        # return
        return [
            obs_rec_list,
            prior_boundary_log_density,
            post_boundary_log_density,
            prior_obs_state_list,
            post_obs_state_list,
            boundary_data_list,
            prior_boundary_list,
            post_boundary_list,
            abs_state_list,
            selected_option,
            onehot_z_list,
            vq_loss_list,
        ]

    def abs_marginal(self, obs_data_list, action_list, seq_size, init_size, n_sample=3):
        #############
        # data size #
        #############
        num_samples = action_list.size(0)
        full_seq_size = action_list.size(1)  # [B, S, C, H, W]

        #######################
        # observation encoder #
        #######################
        enc_obs_list = self.enc_obs(obs_data_list)
        enc_action_list = self.action_encoder(action_list)

        # Shift sequence length dimension forward and 0 out first one
        shifted_enc_actions = torch.roll(enc_action_list, 1, 1)
        mask = torch.ones_like(shifted_enc_actions, device=shifted_enc_actions.device)
        mask[:, 0, :] = 0
        shifted_enc_actions = shifted_enc_actions * mask

        enc_combine_obs_action_list = self.combine_action_obs(
            torch.cat((enc_action_list, enc_obs_list), -1)
        )
        shifted_combined_action_list = self.combine_action_obs(
            torch.cat((shifted_enc_actions, enc_obs_list), -1)
        )

        ######################
        # boundary sampling ##
        ######################
        post_boundary_log_alpha_list = self.post_boundary(shifted_combined_action_list)
        marginal, n = 0, 0

        #############
        # init list #
        #############
        all_codes = []
        all_boundaries = []

        for _ in range(n_sample):
            boundary_data_list, _ = self.boundary_sampler(post_boundary_log_alpha_list)
            boundary_data_list[:, : (init_size + 1), 0] = 1.0
            boundary_data_list[:, : (init_size + 1), 1] = 0.0

            if self._use_min_length_boundary_mask:
                mask = torch.ones_like(boundary_data_list)
                for batch_idx in range(boundary_data_list.shape[0]):
                    reads = torch.where(boundary_data_list[batch_idx, :, 0] == 1)[0]
                    prev_read = reads[0]
                    for read in reads[1:]:
                        if read - prev_read <= 2:
                            mask[batch_idx][read] = 0
                        else:
                            prev_read = read

                boundary_data_list = boundary_data_list * mask
                boundary_data_list[:, :, 1] = 1 - boundary_data_list[:, :, 0]

            boundary_data_list[:, : (init_size + 1), 0] = 1.0
            boundary_data_list[:, : (init_size + 1), 1] = 0.0
            boundary_data_list[:, -init_size:, 0] = 1.0
            boundary_data_list[:, -init_size:, 1] = 0.0

            ######################
            # posterior encoding #
            ######################
            abs_post_fwd_list = []
            abs_post_bwd_list = []
            abs_post_fwd = action_list.new_zeros(
                num_samples, self.abs_belief_size
            ).float()
            abs_post_bwd = action_list.new_zeros(
                num_samples, self.abs_belief_size
            ).float()
            # generating the latent state
            for fwd_t, bwd_t in zip(
                range(full_seq_size), reversed(range(full_seq_size))
            ):
                # forward encoding
                abs_post_fwd = self.abs_post_fwd(
                    enc_combine_obs_action_list[:, fwd_t], abs_post_fwd
                )
                abs_post_fwd_list.append(abs_post_fwd)

                # backward encoding
                bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
                abs_post_bwd = self.abs_post_bwd(
                    enc_combine_obs_action_list[:, bwd_t], abs_post_bwd
                )
                abs_post_bwd_list.append(abs_post_bwd)
                # abs_post_bwd = bwd_copy_data * abs_post_bwd
            abs_post_bwd_list = abs_post_bwd_list[::-1]

            ######################
            # forward transition #
            ######################
            codes = []
            for t in range(init_size, init_size + seq_size):
                #####################
                # (0) get mask data #
                #####################
                read_data = boundary_data_list[:, t, 0].unsqueeze(-1)

                #############################
                # (1) sample abstract state #
                #############################
                _, _, _, onehot_z, z_logit = self.post_abs_state(
                    concat(abs_post_fwd_list[t - 1], abs_post_bwd_list[t])
                )
                log_p = z_logit
                log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
                prob = log_p.exp()
                marginal += (prob * read_data).sum(axis=0)
                n += read_data.sum()
                codes.append(onehot_z)

            all_codes.append(
                torch.stack(codes, axis=1)
            )  # permute such that the shape is (B, S, Z)
            all_boundaries.append(boundary_data_list[:, init_size:-init_size, 0])

        return marginal / n.detach(), all_codes, all_boundaries

    def encoding_cost(self, marginal, codes, boundaries):
        log_marginal = -torch.log(marginal)
        entropy = (log_marginal * marginal).sum()
        num_reads = boundaries.sum(dim=1).mean()
        return entropy * num_reads

    def initial_boundary_state(self, state):
        # Initial padding token
        # Padding action *embedding* is masked out
        enc_action = self.action_encoder(torch.zeros(1).long())
        enc_action = enc_action.squeeze(0) * 0
        padding_state = state * 0
        enc_obs = (
            self.enc_obs(padding_state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        )
        boundary_state = [self.combine_action_obs(torch.cat((enc_action, enc_obs), -1))]

        # First action is set to 0
        enc_action = self.action_encoder(torch.zeros(1).long()).squeeze(0)
        enc_obs = self.enc_obs(state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        boundary_state.append(
            self.combine_action_obs(torch.cat((enc_action, enc_obs), -1))
        )

        return boundary_state

    def z_terminates(self, state, prev_action, boundary_state=None):
        """Returns whether the current z terminates.

        Args:
            state: current state s_t
            prev_action: action a_{t - 1} taken in the previous timestep,
                returned by play_z
            boundary_state: previously returned value from z_terminates or None
                on the first timestep of a new z.

        Returns:
            terminate (bool): True if a new z should be sampled at s_t
            boundary_state: hidden state to be passed back to next call to
                z_terminates.
        """
        # List of combined action and obs embeddings of shape (embed_dim,)
        # The list is of length equal to number of timesteps T current z has
        # been active
        assert boundary_state is not None
        if boundary_state is None:
            boundary_state = []

        # Copy so you don't destructively modify
        boundary_state = list(boundary_state)

        # Dummy batch dimension
        enc_obs = self.enc_obs(state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        enc_action = self.action_encoder(
            torch.tensor(prev_action, device=enc_obs.device)
        )
        # (embed_dim,)
        enc_combine_obs_action = self.combine_action_obs(
            torch.cat((enc_action, enc_obs), 0)
        )

        boundary_state.append(enc_combine_obs_action)
        # (1, T, embed_dim)
        # Needs batch dimension inside of post boundary
        enc_combine_obs_action_list = torch.stack(boundary_state).unsqueeze(0)

        # (2,)
        read_logits = self.post_boundary(enc_combine_obs_action_list)[0, -1]
        terminate = read_logits[0] > read_logits[1]
        return terminate, boundary_state

    def play_z(self, z, state, hidden_state=None, recurrent=False):
        """Returns the action from playing the z at state: a ~ pi(a | s, z).

        Caller should call z_terminates after every call to play_z to determine
        if the same z should be used at the next timestep.

        Args:
            z (int): the option z to use, represented as a single integer (not
                1-hot).
            state: current state s_t

        Returns:
            action (int): a ~ pi(a | z, s_t)
        """
        if hidden_state is None:
            hidden_state = torch.zeros(1, self.abs_belief_size).float()

        # Convert integer
        # No batch dimension here
        # z = self.permitted_zs[z]
        z = self.post_abs_state.z_embedding(z)

        dummy_abs_belief = torch.zeros(self.abs_belief_size, device=z.device)
        abs_feat = self.abs_feat(torch.cat((dummy_abs_belief, z), 0))

        # Add dummy batch dimension before embedding, and then remove, since
        # some embedders require batching
        enc_obs = self.enc_obs(state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        if recurrent:
            hidden_state = self.obs_post_fwd(enc_obs.unsqueeze(0), hidden_state)
            enc_obs = hidden_state.squeeze(0)
        post_obs_state = self.post_obs_state(torch.cat((enc_obs, abs_feat), 0))
        obs_state = post_obs_state
        if self._output_normal:
            obs_state = post_obs_state.mean

        dummy_obs_belief = torch.zeros(abs_feat.shape[0], device=abs_feat.device)
        obs_feat = self.obs_feat(torch.cat((dummy_obs_belief, obs_state), 0))
        return torch.argmax(self.dec_obs(obs_feat)).item(), hidden_state


class EnvModel(nn.Module):
    def __init__(
        self,
        action_encoder,
        encoder,
        decoder,
        belief_size,
        state_size,
        num_layers,
        max_seg_len,
        max_seg_num,
        latent_n,
        rec_coeff=1.0,
        kl_coeff=1.0,
        use_abs_pos_kl=True,
        coding_len_coeff=10.0,
        use_min_length_boundary_mask=False,
        ddo=False,
        output_normal=True
    ):
        super(EnvModel, self).__init__()
        ################
        # network size #
        ################
        self.belief_size = belief_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num
        self.latent_n = latent_n
        self.coding_len_coeff = coding_len_coeff
        self.use_abs_pos_kl = use_abs_pos_kl
        self.kl_coeff = kl_coeff
        self.rec_coeff = rec_coeff

        ##########################
        # baseline related flags #
        ##########################
        self.ddo = ddo

        ###############
        # init models #
        ###############
        # state space model
        self.state_model = HierarchicalStateSpaceModel(
            action_encoder=action_encoder,
            encoder=encoder,
            decoder=decoder,
            belief_size=self.belief_size,
            state_size=self.state_size,
            num_layers=self.num_layers,
            max_seg_len=self.max_seg_len,
            max_seg_num=self.max_seg_num,
            latent_n=self.latent_n,
            use_min_length_boundary_mask=use_min_length_boundary_mask,
            ddo=ddo,
            output_normal=output_normal
        )
        self._output_normal = output_normal

    def forward(self, obs_data_list, action_list, seq_size, init_size, obs_std=1.0):
        ############################
        # (1) run over state model #
        ############################
        [
            obs_rec_list,
            prior_boundary_log_density_list,
            post_boundary_log_density_list,
            prior_obs_state_list,
            post_obs_state_list,
            boundary_data_list,
            prior_boundary_list,
            post_boundary_list,
            abs_state_list,
            selected_option,
            onehot_z_list,
            vq_loss_list,
        ] = self.state_model(obs_data_list, action_list, seq_size, init_size)

        ########################################################
        # (2) compute obs_cost (sum over spatial and channels) #
        ########################################################
        # obs_rec_list: (batch_size, seq_len, action_dim)
        # action_list: (batch_size, seq_len)
        obs_cost = F.cross_entropy(
            obs_rec_list.reshape(-1, obs_rec_list.shape[-1]),
            action_list[:, init_size:-init_size].reshape(-1),
        )

        #######################
        # (3) compute kl_cost #
        #######################
        # compute kl related to states, since we are not using KL for RL
        # setting we avoid the computation
        if self._output_normal:
          kl_obs_state_list = []
          for t in range(seq_size):
             kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
             kl_obs_state_list.append(kl_obs_state.sum(-1))
          kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

          # compute kl related to boundary
          kl_mask_list = post_boundary_log_density_list - prior_boundary_log_density_list
        else:
          kl_obs_state_list = torch.zeros(
              post_obs_state_list[0].shape[0], seq_size)

        ###############################
        # (4) compute encoding length #
        ###############################
        marginal, all_codes, all_boundaries = self.state_model.abs_marginal(
            obs_data_list, action_list, seq_size, init_size
        )
        encoding_length = self.state_model.encoding_cost(
            marginal, onehot_z_list, boundary_data_list.squeeze(-1)
        )

        if self.ddo:
            train_loss = self.rec_coeff * obs_cost.mean() + \
                    self.kl_coeff * kl_mask_list.mean() + \
                    self.coding_len_coeff * encoding_length + \
                    torch.mean(vq_loss_list)
        else:
            train_loss = (
                self.rec_coeff * obs_cost.mean()
                + self.kl_coeff * (kl_obs_state_list.mean() + kl_mask_list.mean())
                + self.coding_len_coeff * encoding_length
                + torch.mean(vq_loss_list)
            )

        pos_obs_state = [x for x in post_obs_state_list]
        if self._output_normal:
            pos_obs_state = [x.mean for x in post_obs_state_list]

        return {
            "rec_data": obs_rec_list,
            "mask_data": boundary_data_list,
            "obs_cost": obs_cost,
            "kl_abs_state": torch.zeros_like(kl_obs_state_list),
            "kl_obs_state": kl_obs_state_list,
            "kl_mask": kl_mask_list,
            "p_mask": prior_boundary_list.mean,
            "q_mask": post_boundary_list.mean,
            "p_ent": prior_boundary_list.entropy(),
            "q_ent": post_boundary_list.entropy(),
            "beta": self.state_model.mask_beta,
            "encoding_length": encoding_length,
            "marginal": marginal.detach().cpu().numpy(),
            "train_loss": train_loss,
            "option_list": selected_option,
            "pos_obs_state": torch.stack(pos_obs_state, axis=1),
            "abs_state": torch.stack(abs_state_list, axis=1),
            "all_boundaries": all_boundaries,
            "vq_loss_list": torch.mean(vq_loss_list).detach(),
        }
