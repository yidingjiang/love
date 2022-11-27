from torch.nn.modules.linear import Linear
from modules import *
from utils import *


class HierarchicalStateSpaceModel(nn.Module):
    def __init__(self,
                 belief_size,
                 state_size,
                 num_layers,
                 max_seg_len,
                 max_seg_num,
                 latent_n=10):
        super(HierarchicalStateSpaceModel, self).__init__()
        ################
        # network size #
        ################
        # abstract level
        self.abs_belief_size = belief_size
        self.abs_state_size = state_size
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
        self.enc_obs = Encoder(feat_size=self.feat_size)
        self.dec_obs = Decoder(input_size=self.obs_feat_size,
                               feat_size=self.feat_size)

        #####################
        # boundary detector #
        #####################
        self.prior_boundary = PriorBoundaryDetector(input_size=self.obs_feat_size)
        self.post_boundary = PostBoundaryDetector(input_size=self.feat_size,
                                                  num_layers=self.num_layers,
                                                  causal=True)

        #####################
        # feature extractor #
        #####################
        self.abs_feat = LinearLayer(input_size=self.abs_belief_size + self.abs_state_size,
                                    output_size=self.abs_feat_size,
                                    nonlinear=nn.Identity())
        self.obs_feat = LinearLayer(input_size=self.obs_belief_size + self.obs_state_size,
                                    output_size=self.obs_feat_size,
                                    nonlinear=nn.Identity())

        #########################
        # belief initialization #
        #########################
        self.init_abs_belief = nn.Identity()
        self.init_obs_belief = nn.Identity()

        #############################
        # belief update (recurrent) #
        #############################
        self.update_abs_belief = RecurrentLayer(input_size=self.abs_state_size,
                                                hidden_size=self.abs_belief_size)
        self.update_obs_belief = RecurrentLayer(input_size=self.obs_state_size + self.abs_feat_size,
                                                hidden_size=self.obs_belief_size)

        #####################
        # posterior encoder #
        #####################
        self.abs_post_fwd = RecurrentLayer(input_size=self.feat_size,
                                           hidden_size=self.abs_belief_size)
        self.abs_post_bwd = RecurrentLayer(input_size=self.feat_size,
                                           hidden_size=self.abs_belief_size)
        self.obs_post_fwd = RecurrentLayer(input_size=self.feat_size,
                                           hidden_size=self.obs_belief_size)

        ####################
        # prior over state #
        ####################
        # self.prior_abs_state = LatentDistribution(input_size=self.abs_belief_size,
        #                                           latent_size=self.abs_state_size)
        self.prior_abs_state = DiscreteLatentDistribution(input_size=self.abs_belief_size,
                                                          latent_size=self.abs_state_size,
                                                          latent_n=latent_n)
        self.prior_obs_state = LatentDistribution(input_size=self.obs_belief_size,
                                                  latent_size=self.obs_state_size)

        ########################
        # posterior over state #
        ########################
        # self.post_abs_state = LatentDistribution(input_size=self.abs_belief_size + self.abs_belief_size,
        #                                          latent_size=self.abs_state_size)
        self.post_abs_state = DiscreteLatentDistribution(
            input_size=self.abs_belief_size + self.abs_belief_size,
            latent_size=self.abs_state_size,
            latent_n=latent_n
        )
        self.post_obs_state = LatentDistribution(input_size=self.obs_belief_size + self.abs_feat_size,
                                                 latent_size=self.obs_state_size)

        self.z_embedding = LinearLayer(input_size=latent_n, output_size=self.abs_state_size)

    # sampler
    def boundary_sampler(self, log_alpha):
        # sample and return corresponding logit
        if self.training:
            log_sample_alpha = gumbel_sampling(log_alpha=log_alpha, temp=self.mask_beta)
        else:
            log_sample_alpha = log_alpha / self.mask_beta

        # probability
        log_sample_alpha = log_sample_alpha - torch.logsumexp(log_sample_alpha, dim=-1, keepdim=True)
        sample_prob = log_sample_alpha.exp()
        sample_data = torch.eye(2, dtype=log_alpha.dtype, device=log_alpha.device)[torch.max(sample_prob, dim=-1)[1]]

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
        near_read_data[:, 1] = - near_read_data[:, 1]
        near_copy_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_copy_data[:, 0] = - near_copy_data[:, 0]

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
            new_log_alpha = over_num * near_copy_data + (1.0 - over_num) * log_alpha_list[:, t]

            # if length is too long (long segment), read
            new_log_alpha = over_len * near_read_data + (1.0 - over_len) * new_log_alpha

            ############
            # (2) save #
            ############
            new_log_alpha_list.append(new_log_alpha)

        # return
        return torch.stack(new_log_alpha_list, dim=1)

    # forward for reconstruction
    def forward(self, obs_data_list, seq_size, init_size):
        #############
        # data size #
        #############
        num_samples = obs_data_list.size(0)
        full_seq_size = obs_data_list.size(1)  # [B, S, C, H, W]

        #######################
        # observation encoder #
        #######################
        enc_obs_list = self.enc_obs(obs_data_list.view(-1, *obs_data_list.size()[2:]))
        enc_obs_list = enc_obs_list.view(num_samples, full_seq_size, -1)  # [B, S, D]

        ######################
        # boundary sampling ##
        ######################
        shifted_enc_obs_list = torch.roll(enc_obs_list, 1, 1)
        mask = torch.ones_like(
                shifted_enc_obs_list, device=shifted_enc_obs_list.device)
        mask[:, 0, :] = 0
        shifted_enc_obs_list *= mask
        post_boundary_log_alpha_list = self.post_boundary(shifted_enc_obs_list)
        # post_boundary_log_alpha_list = self.post_boundary(enc_obs_list)
        boundary_data_list, post_boundary_sample_logit_list = self.boundary_sampler(post_boundary_log_alpha_list)
        boundary_data_list[:, :(init_size + 1), 0] = 1.0
        boundary_data_list[:, :(init_size + 1), 1] = 0.0
        boundary_data_list[:, -init_size:, 0] = 1.0
        boundary_data_list[:, -init_size:, 1] = 0.0

        ######################
        # posterior encoding #
        ######################
        abs_post_fwd_list = []
        abs_post_bwd_list = []
        obs_post_fwd_list = []
        abs_post_fwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_post_bwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        obs_post_fwd = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
        for fwd_t, bwd_t in zip(range(full_seq_size), reversed(range(full_seq_size))):
            # forward encoding
            fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)  # (B, 1)
            abs_post_fwd = self.abs_post_fwd(enc_obs_list[:, fwd_t], abs_post_fwd)
            obs_post_fwd = self.obs_post_fwd(enc_obs_list[:, fwd_t], fwd_copy_data * obs_post_fwd)
            abs_post_fwd_list.append(abs_post_fwd)
            obs_post_fwd_list.append(obs_post_fwd)

            # backward encoding
            bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
            abs_post_bwd = self.abs_post_bwd(enc_obs_list[:, bwd_t], abs_post_bwd)
            abs_post_bwd_list.append(abs_post_bwd)
            abs_post_bwd = bwd_copy_data * abs_post_bwd  # what does this line do?
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

        #######################
        # init state / latent #
        #######################
        abs_belief = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = obs_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_belief = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
        obs_state = obs_data_list.new_zeros(num_samples, self.obs_state_size)

        ######################
        # forward transition #
        ######################
        for t in range(init_size, init_size + seq_size):
            #####################
            # (0) get mask data #
            #####################
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)

            #############################
            # (1) sample abstract state #
            #############################
            if t == init_size:
                abs_belief = self.init_abs_belief(abs_post_fwd_list[t - 1])  # abs_belief is c in the paper
            else:
                abs_belief = read_data * self.update_abs_belief(abs_state, abs_belief) + copy_data * abs_belief
            prior_abs_state = self.prior_abs_state(abs_belief)
            post_abs_state = self.post_abs_state(concat(abs_post_fwd_list[t - 1], abs_post_bwd_list[t]))
            p = post_abs_state.rsample()
            onehot_z_list.append(p)
            sample = self.z_embedding(p)
            abs_state = read_data * sample + copy_data * abs_state
            abs_feat = self.abs_feat(concat(abs_belief, abs_state))
            selected_state = np.argmax(p.detach().cpu().numpy(), axis=-1) # size of batch

            ################################
            # (2) sample observation state #
            ################################
            obs_belief = read_data * self.init_obs_belief(abs_feat) + copy_data * self.update_obs_belief(concat(obs_state, abs_feat), obs_belief)  # this is h
            prior_obs_state = self.prior_obs_state(obs_belief)
            post_obs_state = self.post_obs_state(concat(obs_post_fwd_list[t - 1], abs_feat))  # Look at t-1 to prevent trivial solution
            obs_state = post_obs_state.rsample()
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
            prior_abs_state_list.append(prior_abs_state)
            post_abs_state_list.append(post_abs_state)
            prior_obs_state_list.append(prior_obs_state)
            post_obs_state_list.append(post_obs_state)
            selected_option.append(selected_state)

        # decode all together
        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        obs_rec_list = self.dec_obs(obs_rec_list.view(num_samples * seq_size, -1))
        obs_rec_list = obs_rec_list.view(num_samples, seq_size, *obs_rec_list.size()[-3:])

        # stack results
        prior_boundary_log_alpha_list = torch.stack(prior_boundary_log_alpha_list, dim=1)

        # remove padding
        boundary_data_list = boundary_data_list[:, init_size:(init_size + seq_size)]
        post_boundary_log_alpha_list = post_boundary_log_alpha_list[:, (init_size + 1):(init_size + 1 + seq_size)]
        post_boundary_sample_logit_list = post_boundary_sample_logit_list[:, (init_size + 1):(init_size + 1 + seq_size)]

        # fix prior by constraints
        prior_boundary_log_alpha_list = self.regularize_prior_boundary(prior_boundary_log_alpha_list,
                                                                       boundary_data_list)

        # compute log-density
        prior_boundary_log_density = log_density_concrete(prior_boundary_log_alpha_list,
                                                          post_boundary_sample_logit_list,
                                                          self.mask_beta)
        post_boundary_log_density = log_density_concrete(post_boundary_log_alpha_list,
                                                         post_boundary_sample_logit_list,
                                                         self.mask_beta)

        # compute boundary probability
        prior_boundary_list = F.softmax(prior_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        post_boundary_list = F.softmax(post_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        prior_boundary_list = Bernoulli(probs=prior_boundary_list)
        post_boundary_list = Bernoulli(probs=post_boundary_list)
        boundary_data_list = boundary_data_list[..., 0].unsqueeze(-1)

        # process options
        selected_option = np.stack(selected_option).transpose((1, 0))  # size (B, S)
        onehot_z_list = torch.stack(onehot_z_list, axis=1) # (B, S, Z)

        # return
        return [obs_rec_list,
                prior_boundary_log_density,
                post_boundary_log_density,
                prior_abs_state_list,
                post_abs_state_list,
                prior_obs_state_list,
                post_obs_state_list,
                boundary_data_list,
                prior_boundary_list,
                post_boundary_list,
                selected_option,
                onehot_z_list]

    def abs_marginal(self, obs_data_list, seq_size, init_size, n_sample=3):
        #############
        # data size #
        #############
        num_samples = obs_data_list.size(0)
        full_seq_size = obs_data_list.size(1)  # [B, S, C, H, W]

        #######################
        # observation encoder #
        #######################
        enc_obs_list = self.enc_obs(obs_data_list.view(-1, *obs_data_list.size()[2:]))
        enc_obs_list = enc_obs_list.view(num_samples, full_seq_size, -1)  # [B, S, D]

        ######################
        # boundary sampling ##
        ######################
        shifted_enc_obs_list = torch.roll(enc_obs_list, 1, 1)
        mask = torch.ones_like(
                shifted_enc_obs_list, device=shifted_enc_obs_list.device)
        mask[:, 0, :] = 0
        shifted_enc_obs_list *= mask
        post_boundary_log_alpha_list = self.post_boundary(shifted_enc_obs_list)
        marginal, n = 0, 0

        #############
        # init list #
        #############
        all_codes = []
        all_boundaries = []

        for _ in range(n_sample):
            boundary_data_list, _ = self.boundary_sampler(post_boundary_log_alpha_list)
            boundary_data_list[:, :(init_size + 1), 0] = 1.0
            boundary_data_list[:, :(init_size + 1), 1] = 0.0
            boundary_data_list[:, -init_size:, 0] = 1.0
            boundary_data_list[:, -init_size:, 1] = 0.0

            ######################
            # posterior encoding #
            ######################
            abs_post_fwd_list = []
            abs_post_bwd_list = []
            obs_post_fwd_list = []
            abs_post_fwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
            abs_post_bwd = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
            obs_post_fwd = obs_data_list.new_zeros(num_samples, self.obs_belief_size)
            # generating the latent state
            for fwd_t, bwd_t in zip(range(full_seq_size), reversed(range(full_seq_size))):
                # forward encoding
                fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)
                abs_post_fwd = self.abs_post_fwd(enc_obs_list[:, fwd_t], abs_post_fwd)
                obs_post_fwd = self.obs_post_fwd(enc_obs_list[:, fwd_t], fwd_copy_data * obs_post_fwd)
                abs_post_fwd_list.append(abs_post_fwd)
                obs_post_fwd_list.append(obs_post_fwd)

                # backward encoding
                bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
                abs_post_bwd = self.abs_post_bwd(enc_obs_list[:, bwd_t], abs_post_bwd)
                abs_post_bwd_list.append(abs_post_bwd)
                abs_post_bwd = bwd_copy_data * abs_post_bwd
            abs_post_bwd_list = abs_post_bwd_list[::-1]

            #######################
            # init state / latent #
            #######################
            abs_belief = obs_data_list.new_zeros(num_samples, self.abs_belief_size)
            abs_state = obs_data_list.new_zeros(num_samples, self.abs_state_size)

            ######################
            # forward transition #
            ######################
            codes = []
            for t in range(init_size, init_size + seq_size):
                #####################
                # (0) get mask data #
                #####################
                read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
                copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)

                #############################
                # (1) sample abstract state #
                #############################
                if t == init_size:
                    abs_belief = self.init_abs_belief(abs_post_fwd_list[t - 1])
                else:
                    abs_belief = read_data * self.update_abs_belief(abs_state, abs_belief) + copy_data * abs_belief
                # distribution
                post_abs_state = self.post_abs_state(concat(abs_post_fwd_list[t - 1], abs_post_bwd_list[t]))
                abs_state = read_data * self.z_embedding(post_abs_state.rsample()) + copy_data * abs_state
                log_p = post_abs_state.log_p
                log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
                prob = log_p.exp()
                marginal += (prob * read_data).sum(axis=0)
                n += read_data.sum()
                codes.append(post_abs_state.rsample().detach())

            all_codes.append(torch.stack(codes, axis=1)) # permute such that the shape is (B, S, Z)
            all_boundaries.append(torch.argmax(boundary_data_list, dim=-1).detach()[:, init_size:-init_size])

        return marginal / n.detach(), all_codes, all_boundaries

    def encoding_cost(self, marginal, codes, boundaries):
        log_marginal = -torch.log(marginal)
        entropy = (log_marginal * marginal).sum()
        num_reads = boundaries.sum(dim=1).mean()
        return entropy * num_reads

    # generation forward
    def full_generation(self, init_data_list, seq_size):
        # eval mode
        self.eval()

        ########################
        # initialize variables #
        ########################
        num_samples = init_data_list.size(0)
        init_size = init_data_list.size(1)

        ####################
        # forward encoding #
        ####################
        abs_post_fwd = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        for t in range(init_size):
            abs_post_fwd = self.abs_post_fwd(self.enc_obs(init_data_list[:, t]), abs_post_fwd)

        ##############
        # init state #
        ##############
        abs_belief = init_data_list.new_zeros(num_samples, self.abs_belief_size)
        abs_state = init_data_list.new_zeros(num_samples, self.abs_state_size)
        obs_belief = init_data_list.new_zeros(num_samples, self.obs_belief_size)
        obs_state = init_data_list.new_zeros(num_samples, self.obs_state_size)

        #############
        # init list #
        #############
        obs_rec_list = []
        boundary_data_list = []
        option_list = []

        ######################
        # forward transition #
        ######################
        read_data = init_data_list.new_ones(num_samples, 1)
        copy_data = 1 - read_data
        for t in range(seq_size):
            #############################
            # (1) sample abstract state #
            #############################
            if t == 0:
                abs_belief = self.init_abs_belief(abs_post_fwd)
            else:
                abs_belief = read_data * self.update_abs_belief(abs_state, abs_belief) + copy_data * abs_belief
            
            p = self.prior_abs_state(abs_belief).rsample()
            sample = self.z_embedding(p)
            abs_state = read_data * sample + copy_data * abs_state
            abs_feat = self.abs_feat(concat(abs_belief, abs_state))

            ################################
            # (2) sample observation state #
            ################################
            obs_belief = read_data * self.init_obs_belief(abs_feat) + copy_data * self.update_obs_belief(concat(obs_state, abs_feat), obs_belief)
            obs_state = self.prior_obs_state(obs_belief).rsample()
            obs_feat = self.obs_feat(concat(obs_belief, obs_state))

            ##########################
            # (3) decode observation #
            ##########################
            obs_rec = self.dec_obs(obs_feat)

            ############
            # (4) save #
            ############
            obs_rec_list.append(obs_rec)
            boundary_data_list.append(read_data)
            option_list.append(np.argmax(p.detach().cpu().numpy(), -1))

            ###################
            # (5) sample mask #
            ###################
            prior_boundary = self.boundary_sampler(self.prior_boundary(obs_feat))[0]
            read_data = prior_boundary[:, 0].unsqueeze(-1)
            copy_data = prior_boundary[:, 1].unsqueeze(-1)

        # stack results
        obs_rec_list = torch.stack(obs_rec_list, dim=1)
        boundary_data_list = torch.stack(boundary_data_list, dim=1)
        option_list = np.stack(option_list, axis=1)
        return obs_rec_list, boundary_data_list, option_list


class EnvModel(nn.Module):
    def __init__(self,
                 belief_size,
                 state_size,
                 num_layers,
                 max_seg_len,
                 max_seg_num,
                 use_abs_pos_kl=True,
                 coding_len_coeff=10.0):
        super(EnvModel, self).__init__()
        ################
        # network size #
        ################
        self.belief_size = belief_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num
        self.coding_len_coeff = coding_len_coeff
        self.use_abs_pos_kl = use_abs_pos_kl

        ###############
        # init models #
        ###############
        # state space model
        self.state_model = HierarchicalStateSpaceModel(belief_size=self.belief_size,
                                                       state_size=self.state_size,
                                                       num_layers=self.num_layers,
                                                       max_seg_len=self.max_seg_len,
                                                       max_seg_num=self.max_seg_num)

    def forward(self, obs_data_list, seq_size, init_size, obs_std=1.0):
        ############################
        # (1) run over state model #
        ############################
        [obs_rec_list,
         prior_boundary_log_density_list,
         post_boundary_log_density_list,
         prior_abs_state_list,
         post_abs_state_list,
         prior_obs_state_list,
         post_obs_state_list,
         boundary_data_list,
         prior_boundary_list,
         post_boundary_list,
         selected_option,
         onehot_z_list] = self.state_model(obs_data_list, seq_size, init_size)

        ########################################################
        # (2) compute obs_cost (sum over spatial and channels) #
        ########################################################
        obs_target_list = obs_data_list[:, init_size:-init_size]
        obs_cost = - Normal(obs_rec_list, obs_std).log_prob(obs_target_list)
        obs_cost = obs_cost.sum(dim=[2, 3, 4])

        #######################
        # (3) compute kl_cost #
        #######################
        # compute kl related to states
        kl_abs_state_list = []
        kl_obs_state_list = []
        for t in range(seq_size):
            # read flag
            read_data = boundary_data_list[:, t].detach()

            # kl divergences (sum over dimension)
            # kl_abs_state = kl_divergence(post_abs_state_list[t], prior_abs_state_list[t]) * read_data
            kl_abs_state = kl_categorical(post_abs_state_list[t], prior_abs_state_list[t], mask_a=not self.use_abs_pos_kl) * read_data
            kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
            # kl_abs_state_list.append(kl_abs_state.sum(-1))
            kl_abs_state_list.append(kl_abs_state)
            kl_obs_state_list.append(kl_obs_state.sum(-1))
        kl_abs_state_list = torch.stack(kl_abs_state_list, dim=1)
        kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

        # compute kl related to boundary
        kl_mask_list = (post_boundary_log_density_list - prior_boundary_log_density_list)

        ###############################
        # (4) compute encoding length #
        ###############################
        marginal, all_codes, all_boundaries = self.state_model.abs_marginal(obs_data_list, seq_size, init_size)
        all_codes = torch.cat(all_codes, dim=0)
        all_boundaries = torch.cat(all_boundaries, dim=0)
        encoding_length = self.state_model.encoding_cost(marginal, onehot_z_list, boundary_data_list.squeeze(-1))

        # return
        return {'rec_data': obs_rec_list,
                'mask_data': boundary_data_list,
                'obs_cost': obs_cost,
                'kl_abs_state': kl_abs_state_list,
                'kl_obs_state': kl_obs_state_list,
                'kl_mask': kl_mask_list,
                'p_mask': prior_boundary_list.mean,
                'q_mask': post_boundary_list.mean,
                'p_ent': prior_boundary_list.entropy(),
                'q_ent': post_boundary_list.entropy(),
                'beta': self.state_model.mask_beta,
                'encoding_length': encoding_length,
                'marginal': marginal.detach().cpu().numpy(),
                'train_loss': obs_cost.mean() + kl_abs_state_list.mean() + kl_obs_state_list.mean() + kl_mask_list.mean() + self.coding_len_coeff * encoding_length,
                'option_list': selected_option,
                }

    def full_generation(self, init_obs_list, seq_size):
        return self.state_model.full_generation(init_obs_list, seq_size)
