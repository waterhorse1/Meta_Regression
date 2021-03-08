import arguments
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
    else:
        if args.maml:
            logger = maml.run(args, log_interval=100, rerun=True)

    