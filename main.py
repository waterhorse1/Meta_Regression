import arguments
import ella
import cavia
import cavia_backup
import cavia_do
import cavia_ori
import cavia_recon
import cavia_rnn
import maml


if __name__ == '__main__':

    args = arguments.parse_args()
    #cavia.run(args, log_interval=100, rerun=True)
    #ella.run(args, log_interval=100, rerun=True)
    #exit()
    #maml.run(args, log_interval=100, rerun=True)
    
    if args.test:
        if args.maml:
            logger = maml.test(args)
        elif args.cavia_all:
            logger = cavia_ori.test(args)
        else:
            logger = cavia.test(args)
    else:
        if args.maml:
            logger = maml.run(args, log_interval=100, rerun=True)
        elif args.cavia_all:
            logger = cavia_ori.run(args, log_interval=500, rerun=True)
        elif args.cavia_recon:
            logger = cavia_rnn.run(args, log_interval=100, rerun=True)
        else:
            logger = cavia.run(args, log_interval=500, rerun=True)
        #logger = cavia.run(args, log_interval=100, rerun=True)
        #logger = cavia_backup.run(args, log_interval=100, rerun=True)
    