def init_datatable_mode():
    import pandas as pd
    from IPython.core.display import display, Javascript

    # configure path to the datatables library using requireJS
    # that way the library will become globally available
    display(
        Javascript(
            """
        require.config({
            paths: {
                DT: '//cdn.datatables.net/1.10.19/js/jquery.dataTables.min',
                JQ: '//code.jquery.com/jquery-3.3.1.min'
            }
        });
        require(["JQ"], function (JQ) {
            if ($("head link#datatable-css-fds82j1").length == 0) {
                $('head').append('<link rel="stylesheet" id="datatable-css-fds82j1" type="text/css" href="//cdn.datatables.net/1.10.19/css/jquery.dataTables.min.css">');
            }
        });
    """
        )
    )

    def _repr_datatable_(self):
        """Return DataTable representation of pandas DataFrame."""
        # classes for dataframe table (optional)
        classes = ["table", "table-striped", "table-bordered"]

        # create table DOM
        script = f'$(element).html(`{self.to_html(index=True, classes=classes, border=0, justify="left")}`);\n'

        # execute jQuery to turn table into DataTable
        script += """
            require(["DT", "JQ"], function(DT) {
                $(document).ready( () => {
                    // Turn existing table into datatable
                    $(element).find("table.dataframe").DataTable();
                })
            });
        """

        return script

    pd.DataFrame._repr_javascript_ = _repr_datatable_
