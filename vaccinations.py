import os

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
# import wpcc

from data import PROJECT_RAW_DATA_FOLDER, PROJECT_ROOT_FOLDER, PROJECT_PROCESSED_DATA_FOLDER
from enums.EColor import EColor


if __name__ == '__main__':
    # REGULAR VNR GRAPH
    ####################################################################################################################
    full_graphs = False
    release = False
    x_log_scale = True
    cities_data_date = '2021-01-12'
    cases_signal = "Active_cases"  # {"Active_cases", "Cumulative_verified_cases"}
    size_key = "population_60+"
    x_data_date = "12/1/21"
    y_data_date = "12/1/21"
    x_key = "{} per 10,000 people - {}".format(cases_signal.replace('_', ' '), x_data_date)
    y_key = "60+ Vaccination [%] - {}".format(y_data_date)
    color_key = "VNR"
    color_scale = [EColor.GREEN,
                   EColor.ORANGE,
                   EColor.RED]

    # Cases data - x
    cities_data_path = os.path.join(PROJECT_RAW_DATA_FOLDER,
                                    'corona_city_table_ver_0035.csv')
    cities_df = pd.read_csv(cities_data_path, encoding='utf-8-sig')

    # Vaccination data - y
    vaccination_data_path = os.path.join(PROJECT_RAW_DATA_FOLDER,
                                         'vaccinated_by_age_2021_01_13.csv')
    # pre-processing vaccination data
    vaccination_df = pd.read_csv(vaccination_data_path, encoding='utf-8-sig')

    vaccination_df["drop_row"] = False
    for index, row in vaccination_df.iterrows():
        cnt = int(row["60-69"] == '< 15') + \
              int(row["70-79"] == '< 15') + \
              int(row["80-89"] == '< 15') + \
              int(row["90+"] == '< 15')
        if cnt > 1:
            vaccination_df.loc[index, "drop_row"] = True
    vaccination_df = vaccination_df[vaccination_df.drop_row == False]
    vaccination_df = vaccination_df.drop("drop_row", 1)

    vaccination_df.replace('< 15', 1, inplace=True)
    vaccination_df.replace(np.nan, -100, inplace=True)
    vaccination_df.set_index('City_Name', inplace=True)
    vaccination_df = vaccination_df.astype('float').astype('int')
    vaccination_df.replace(-100, np.nan, inplace=True)
    vaccination_df.reset_index(inplace=True)

    # Population data
    population_path = os.path.join(PROJECT_RAW_DATA_FOLDER,
                                   'israel-city_pop.csv')

    # Social-Economic Rank data
    rank_data_path = os.path.join(PROJECT_RAW_DATA_FOLDER,
                                  'se_index_2017.csv')
    rank_df = pd.read_csv(rank_data_path, encoding='utf-8-sig')
    rank_df = rank_df.dropna()

    # rank_df["Rank"] = rank_df["Rank"].astype('float').astype('int')
    rank_df["Rank"] = rank_df["Rank"].astype('float')

    pop_df = pd.read_csv(population_path, encoding='utf-8-sig')

    # pre-processing population data
    pop_df.replace('..', -100, inplace=True)
    pop_df = pop_df.astype('float').astype('int')
    pop_df.replace(-100, 0, inplace=True)

    # Calculating 60+ population
    pop_df["population_60+"] = pop_df["pop_64-60"] + \
                               pop_df["pop_69-65"] + \
                               pop_df["pop_74-70"] + \
                               pop_df["pop_79-75"] + \
                               pop_df["pop_84-80"] + \
                               pop_df["pop_+85"]

    # Calculating 60+ vaccinations
    vaccination_df["vaccination_60+"] = vaccination_df["60-69"].fillna(0) + \
                                        vaccination_df["70-79"].fillna(0) + \
                                        vaccination_df["80-89"].fillna(0) + \
                                        vaccination_df["90+"].fillna(0)

    # Filtering relevant vaccination data
    vaccination_60_plus = vaccination_df[["City_Name", "vaccination_60+"]]
    vaccination_60_plus = vaccination_60_plus.replace(np.nan, 0)

    # Filtering relevant population data
    pop_df = pop_df[["UID", "population_60+", "total_pop"]]

    bubble_table = cities_df[cities_df['Date'] == cities_data_date].copy()

    # Pre-processing relevant cities data
    bubble_table["Cumulative_verified_cases"].replace('<15', 1, inplace=True)
    bubble_table["Cumulative_verified_cases"] = bubble_table["Cumulative_verified_cases"].astype('float').astype('int')

    bubble_table["Cumulated_recovered"].replace('<15', 1, inplace=True)
    bubble_table["Cumulated_recovered"] = bubble_table["Cumulated_recovered"].astype('float').astype('int')

    bubble_table["Cumulated_deaths"].replace('<15', 1, inplace=True)
    bubble_table["Cumulated_deaths"] = bubble_table["Cumulated_deaths"].astype('float').astype('int')

    bubble_table["Cumulated_vaccinated"].replace('<15', 1, inplace=True)
    bubble_table["Cumulated_vaccinated"] = bubble_table["Cumulated_vaccinated"].astype('float').astype('int')

    # Calculating active cases
    bubble_table["Active_cases"] = bubble_table["Cumulative_verified_cases"] - \
                                   bubble_table["Cumulated_recovered"] - \
                                   bubble_table["Cumulated_deaths"]

    # Fixing active cases mistakes
    bubble_table.loc[bubble_table["Active_cases"] < 0, "Active_cases"] = 0

    # Merging population data
    bubble_table = bubble_table.merge(pop_df, left_on="City_Code", right_on="UID")

    # Merging vaccination data
    bubble_table = bubble_table.merge(vaccination_60_plus, left_on="City_Name", right_on="City_Name")

    # Merging Social-economic rank data
    bubble_table = bubble_table.merge(rank_df, on="UID")

    # Calculating x
    bubble_table[x_key] = \
        bubble_table[cases_signal] / (bubble_table["total_pop"] / 10000)

    # Calculating y
    bubble_table[y_key] = \
        (bubble_table["vaccination_60+"] / (bubble_table["population_60+"])) * 100

    # Calculating metric (color)
    bubble_table["VNR"] = bubble_table[x_key] / \
                          bubble_table[y_key]

    # Calculating color range
    low_color = bubble_table[color_key].quantile(0.2)
    high_color = bubble_table[color_key].quantile(0.8)

    show_table = bubble_table.copy()
    # show_table.dropna(subset=['VNR'], inplace=True)  # unneeded

    if not full_graphs:
        # remove cities where 60+ population is less then 1000
        show_table.drop(show_table.loc[show_table['population_60+'] < 1000].index, inplace=True)
        if cases_signal is "Active_cases":
            chosen_x_range = [20, 300]
        elif cases_signal is "Cumulative_verified_cases":
            chosen_x_range = [180, 1700]
        else:
            print("x_range was not specified. using None.")
        chosen_y_range = [21, 91]
    else:
        chosen_x_range = None
        chosen_y_range = None

    # remove text of cities with total population less then 30k
    show_table["text"] = show_table["name_e"]
    show_table.loc[(show_table.total_pop < 30000), "text"] = ""

    fig = px.scatter(show_table,
                     x=x_key,
                     y=y_key,
                     size=size_key,
                     color=color_key,
                     hover_name="name_e",
                     hover_data={'Cumulated_vaccinated': ':.0f',
                                 'population_60+': ':.0f',
                                 'vaccination_60+': ':.0f'},
                     log_x=x_log_scale,
                     opacity=None,  # between 0 to 1
                     range_x=chosen_x_range,
                     range_y=chosen_y_range,
                     size_max=120,
                     height=1000,
                     text="text",
                     # template='plotly_dark',
                     color_continuous_scale=color_scale,
                     range_color=[low_color, high_color])

    # Set titles
    fig.update_layout(title="<b>60+ Vaccination vs. {} <br>"
                            "Size = {}, Color = Vaccination Need Ratio (VNR)<b>"
                      .format(x_key.replace('_', ' ').capitalize().split('-')[0],
                              size_key.replace('_', ' ').capitalize()),
                      title_x=0.5,
                      font=dict(
                          size=11,
                      ),
                      xaxis_title='<b>' + "{}".format(
                          x_key.replace('_', ' ')) + ' (Logarithmic Scale)' if x_log_scale else '' + '<b>',
                      yaxis_title='<b>' + y_key + '<b>',
                      margin=dict(r=0, l=0, b=0, t=150),
                      template="plotly_white"
                      )

    if release:
        # Release hover template
        fig.data[0].update(hovertemplate='<b>%{hovertext}</b><br><br>' +
                                         'Vaccination Need Ratio' + ' = %{marker.color:.3f}<br>')
    else:
        # Full hover template
        fig.data[0].update(hovertemplate='<b>%{hovertext}</b><br><br>' +
                                         'Accumulated Cases per 10,000' + ' = %{x: .0f}<br>' +
                                         '60+ Vaccination' + ' = %{y: .0f}%<br>' +
                                         '<br>' +
                                         'Accumulated 60+ Vaccinated' + ' = %{customdata[2]: .0f}<br>' +
                                         'Population over 60' + ' = %{customdata[1]: .0f}<br>' +
                                         'Accumulated Vaccinated' + ' = %{customdata[0]: .0f}<br>' +
                                         'Accumulated Cases' + ' = %{marker.size}<br>' +
                                         '<br>' +
                                         'Vaccination Need Ratio' + ' = %{marker.color:.3f}<br>')

    # show graph
    fig.show()

    # Create results directory
    results_directory = os.path.join(PROJECT_ROOT_FOLDER, 'results')
    os.makedirs(results_directory, exist_ok=True)

    # Save to html
    html_file_name = '60+-Vaccination-vs-{}{}.html'.format(
        x_key.replace('_', ' ').capitalize().split('-')[0].replace(' ', '-'),
        'release' if release else '').replace(',', '')

    html_save_path = os.path.join(results_directory, html_file_name)
    with open(html_save_path, 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Save high quality image
    # png
    required_extensions = ['.png', '.pdf', '.svg']
    for extension in required_extensions:
        x_axis_name = x_key.replace('_', ' ').capitalize().split('-')[0]
        save_path = os.path.join(results_directory,
                                 '60+ Vaccination vs. {}{}'.format(x_axis_name, extension))
        fig.write_image(save_path,
                        width=1920,
                        height=1080
                        )

    # END OF REGULAR VNR
    ####################################################################################################################

    # SE RANK - VNR GRAPH
    ####################################################################################################################

    if not full_graphs:
        chosen_x_range = [1, 255]
        chosen_y_range = [21, 92]
    else:
        chosen_x_range = None
        chosen_y_range = None

    fig = px.scatter(show_table,
                     x="Rank",
                     y=y_key,
                     size=size_key,
                     color=color_key,
                     hover_name="name_e",
                     hover_data={'Cumulated_vaccinated': ':.0f',
                                 'population_60+': ':.0f',
                                 'vaccination_60+': ':.0f'},
                     range_x=chosen_x_range,
                     range_y=chosen_y_range,
                     size_max=120,
                     text="text",
                     color_continuous_scale=color_scale,
                     range_color=[low_color, high_color])

    chosen_x_title = 'Socioeconomic Rank'
    chosen_x_axes_title = 'Socioeconomic Rank - 2017'
    fig.update_layout(title="<b>60+ Vaccination vs. {}<br>"
                            "Size = {}, Color = Vaccination Need Ratio (VNR)<b>"
                      .format(chosen_x_title,
                              size_key.replace('_', ' ').capitalize()),
                      title_x=0.5,
                      xaxis_title='<b>{}<b>'.format(chosen_x_axes_title),
                      yaxis_title='<b>' + y_key + '<b>',

                      margin=dict(r=0, l=0, b=0, t=100),
                      template="plotly_white"
                      )

    fig.data[0].update(hovertemplate='<b>%{hovertext}</b><br><br>' +
                                     chosen_x_title + ' = %{x: .0f}<br>' +
                                     '60+ Vaccination' + ' = %{y: .0f}%<br>' +
                                     '<br>' +
                                     'Accumulated 60+ Vaccinated' + ' = %{customdata[2]: .0f}<br>' +
                                     'Population over 60' + ' = %{customdata[1]: .0f}<br>' +
                                     'Accumulated Vaccinated' + ' = %{customdata[0]: .0f}<br>' +
                                     'Accumulated Cases' + ' = %{marker.size}<br>' +
                                     '<br>' +
                                     'Vaccination Need Ratio' + ' = %{marker.color:.3f}<br>')

    ses_html_file_name = '60+ Vaccination vs. {}.html'.format(chosen_x_title)
    ses_html_save_path = os.path.join(results_directory, ses_html_file_name)
    fig.write_html(ses_html_save_path, full_html=False, include_plotlyjs='cdn')

    fig.show()

    # Save high quality image
    for extension in required_extensions:
        file_name = '60+ Vaccination vs. Socioeconomic Rank'
        save_path = os.path.join(results_directory,
                                 '{}{}'.format(file_name, extension))
        fig.write_image(save_path,
                        width=1920,
                        height=1080
                        )

    def pearsonr_pval(x, y):
        return pearsonr(x, y)[1]


    def spearmanr_pval(x, y):
        return spearmanr(x, y)[1]


    def m(x, w):
        """Weighted Mean"""
        return np.sum(x * w) / np.sum(w)


    def cov(x, y, w):
        """Weighted Covariance"""
        return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


    def corr(x, y, w):
        """Weighted Correlation"""
        return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

    def generate_confidence_interval_data(repetitions, x, y, w, correlations_data, correlation_func):
        ret = []
        for i in range(repetitions):
            samples_df = correlations_data.sample(len(correlations_data[x]), replace=True, random_state=i)
            ret.append(correlation_func(samples_df[x], samples_df[y], samples_df[w]))
        return ret

    def calc_ci(data, alpha, repetitions):
        data.sort()
        trim_val = int((1 - alpha) * repetitions / 2)
        del data[len(data) - trim_val:]
        del data[:trim_val]
        return min(data), max(data)

    def calc_correlations(x, y, w):
        corr_data = bubble_table[[x, y, w]]
        alpha = 0.95
        repetitions = 10000

        # # Calculate weighted correlation - 1st method
        # weighted_correlation_1st_method = wpcc.wpearson(corr_data[x], corr_data[y], corr_data[w])
        # ci_data = generate_confidence_interval_data(repetitions, x, y, w, corr_data, wpcc.wpearson)
        # ci_1st_method = calc_ci(ci_data, alpha, repetitions)

        # Calculate weighted correlation - 2nd method
        weighted_correlation_2nd_method = corr(corr_data[x], corr_data[y], corr_data[w])
        ci_data = generate_confidence_interval_data(repetitions, x, y, w, corr_data, corr)
        ci_2nd_method = calc_ci(ci_data, alpha, repetitions)

        # Calculate the pearson correlation
        pearson_correlation = corr_data[y].corr(corr_data[x], method='pearson')
        pearson_p_value = corr_data[y].corr(corr_data[x], method=pearsonr_pval)

        # Calculate the spearman correlation
        spearman_correlation = corr_data[y].corr(corr_data[x], method='spearman')
        spearman_p_value = corr_data[y].corr(corr_data[x], method=spearmanr_pval)

        # print("Weighted Correlation - 1st method: {}".format(weighted_correlation_1st_method) +
        #       "; Confidence Interval (alpha={}) : {}".format(alpha, ci_1st_method))
        print("Weighted Correlation - 2nd method: {}".format(weighted_correlation_2nd_method) +
              "; Confidence Interval: (alpha={}) : {}".format(alpha, ci_2nd_method))
        print("Pearson Correlation: {} +- {}".format(pearson_correlation, pearson_p_value))
        print("Spearman Correlation: {} +- {}".format(spearman_correlation, spearman_p_value))
        print()

    ###################
    # Calc correlations
    ###################

    # save original keys for later
    x_original = x_key
    y_original = y_key

    # Calc weights of cities by their relative 60+ population size
    total_old_population = bubble_table["population_60+"].sum()
    w_key = "weight"
    bubble_table[w_key] = (bubble_table["population_60+"] / total_old_population) * 100

    # 1) 60+ Vaccinations % ~ Cases per 10000
    print("(1) - 60+ Vaccinations % ~ {}".format(x_key.replace('_', ' ')))
    calc_correlations(x_key, y_key, w_key)

    # 2) 60+ Vaccinations % ~ Socioeconomic Rank
    x_key = "Rank"
    print("(2) - 60+ Vaccinations % ~ Socioeconomic Rank")
    calc_correlations(x_key, y_key, w_key)

    # 3) Socioeconomic Rank ~ Cases per 10000
    x_key = x_original
    y_key = "Rank"
    print("(3) - Socioeconomic Rank ~ {}".format(x_key.replace('_', ' ')))
    calc_correlations(x_key, y_key, w_key)

    # 4) Socioeconomic Rank ~ VNR
    x_key = "VNR"
    y_key = "Rank"
    print("(4) - Socioeconomic Rank ~ {}".format(x_key.replace('_', ' ')))
    calc_correlations(x_key, y_key, w_key)

    # Correlogram
    corr_table = bubble_table.copy()
    corr_table = corr_table[[x_original, y_original, 'Rank', 'VNR']]
    sns.pairplot(corr_table, kind="reg")
    # plt.show()  # need to save manually to .png, .pdf, .svg after setting aspect ratio.

    # # save to file
    # plt.savefig('correlogram.png')
    # plt.savefig('correlogram.pdf')
    # plt.savefig('correlogram.svg')

    # save raw data to csv
    processed_data_save_path = os.path.join(PROJECT_PROCESSED_DATA_FOLDER, 'processed_data_merged.csv')
    bubble_table.to_csv(processed_data_save_path, encoding='utf-8-sig', index=False)
