# from data import cfg, set_cfg, set_dataset

# def yoloactEval(input_folder, output_folder):
#     #configing the net

#     args.trained_model = SavePath.get_interrupt('yolact/weights')

#     with torch.no_grad():
#         if not os.path.exists('results'):
#             os.makedirs('results')

        
#         torch.set_default_tensor_type('torch.FloatTensor')
    
#         dataset = None  

#         print('Loading model...', end='')
#         net = Yolact()
#         net.load_weights(args.trained_model)
#         net.eval()
#         print(' Done.')

#         #evaluate(net, dataset)

#         evalimages(net, inp, out)




os.system('python eval.py --trained_model interrupt --config yolact_resnet50_foot_pron_config --score_threshold 0.15 --top_k 1 --output_coco_json --images ../outp  uts/:../predictions/')
