import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import torch
import numpy as np


def pca_analysis_latents(model, inputs, behv, behv_list, save_result_path, pca_numb_comp, title):
    in_ana=torch.tensor(inputs, dtype=torch.float32).to(model.device)
    in_ana=torch.swapaxes(in_ana,0,1)
    out_true_ana=behv
    _,_,out_hat=model.encoder_neural(in_ana)
    out_hat=torch.swapaxes(out_hat,0,1).detach().cpu().numpy()
    from sklearn import decomposition
    pca = decomposition.PCA(n_components=pca_numb_comp)
    out_hat_ana=np.zeros((out_hat.shape[0], out_hat.shape[1],pca_numb_comp ))
    for ii in range(out_hat_ana.shape[0]):
        out_hat_ana[ii,:,:]= pca.fit_transform(out_hat[ii,:,:])
    corr_df=pd.DataFrame()
    corr_df[behv_list]=out_true_ana.mean(axis=0)
    corr_df[['pca'+str(i) for i in range(pca_numb_comp)]]=out_hat_ana.mean(axis=0)
    corr_matrix = corr_df.corr()
    sns.heatmap(np.abs(corr_matrix), annot=True)
    plt.title(title)
    plt.savefig(save_result_path + title + '.png')
    plt.savefig(save_result_path + title + '.svg',
                format='svg')
    plt.close()
def visualize_confMatrix(save_result_path, data,labels, title ):
    df_cm = pd.DataFrame(np.log(data+1), columns=labels, index=labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap="Blues", annot=False, annot_kws={"size": 16})
    plt.title(title)
    plt.savefig(save_result_path + title + '.png')
    plt.savefig(save_result_path + title+ '.svg',
                format='svg')
    plt.close()

def plot_sample_evaluate(save_result_path,model, iterator, label,no_LM=False):
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src ,trg= batch
            src = torch.swapaxes(src, 0, 1)
            trg = torch.swapaxes(trg, 0, 1)
            output,output_neural,_ = model(src, trg, 0,False)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
        for ii in np.arange(1,src.shape[1]):
            fig, ax =plt.subplots( 2,1, sharex=True, sharey=True)

            ax[0].stem(trg[:,ii].detach().cpu().numpy().squeeze(), 'b')
            ax[0].set_title('true sentence sample-'+str(ii)+'-result for ' + label)
            if no_LM:
                ax[1].stem(output_neural.detach().cpu().numpy().argmax(axis=-1)[:, ii].squeeze(), 'r')
            else:

                ax[1].stem(output.detach().cpu().numpy().argmax(axis=-1)[:, ii].squeeze(), 'r')
            ax[1].set_title('predicted sentence sample'+str(ii)+' result for ' + label)
            plt.savefig(save_result_path + 'predicted-sentence-sample'+str(ii)+' result for ' + label + '.png')
            plt.savefig(save_result_path + 'predicted-sentence-sample'+str(ii)+' result for ' + label + '.svg', format='svg')
            plt.close()
