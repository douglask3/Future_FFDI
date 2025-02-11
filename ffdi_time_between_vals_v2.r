graphics.off()
country = c('Global' = '', Australia = 'Australia', 'Brazil' = 'Brazil', 'USA' = 'United States of America')
files = paste0("outputs/cal_time-", country, ".csv")
#dat = read.csv(file)
 
rcps = c("2.6" = "rcp2_6", "8.5" = "rcp8_5")
degs = c("Baseline" = -1, "None" = 0, "1_5" = 1.5, "2_deg" = 2, "4_deg" = 4)
cols = rev(c('#FFFFFF', '#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026'))
diff = FALSE
xlog = TRUE
tmin = 1/24
 
cal_plot_data <- function(rcp, deg, file) {
        dat = read.csv(file)
	index = which(grepl(rcp, colnames(dat)) & grepl(deg, colnames(dat)))
	index0 = which(grepl(rcp, colnames(dat)) & grepl("Baseline", colnames(dat)))
	out = dat[index] 
	
	if (deg != 'Baseline') out = out + dat[index0]
	#else {
		out = out[nrow(out):1,]
		#out = t(sapply(2:nrow(out), function(i) out[i,] + apply(out[1:(i-1),], 2, sum)))
		#out = apply(out, 1, function(i) 365*unlist(i)/unlist(out[nrow(out),]))
		#out = t(out)
		#out = out
	#}	
	out = as.matrix(out)
	out = rbind(out, 365)
        #out[] = 365/6
        #for (i in 1:nrow(out)) out[i,] = out[i,]*i
        #browser()
	return(out)
}

cal_country_dat <- function(file) 
    dat = lapply(names(degs), function(deg) lapply(rcps, cal_plot_data, deg, file))

dats = lapply(files, cal_country_dat)
plot_rcpDeg <- function(rcp_i, deg_j, dat) {
        
	if (rcp_i == 1 && deg_j == 1) return()
#	if (rcp_i == 2 && deg_j == 2) return()
	y = 1.5*((deg_j-1) * (2.5+length(rcps)) + 2*rcp_i - 3)
	yl = rep(y, 2)
	tdat = dat[[deg_j]][[rcp_i]]
        
	tdat = rbind(0, tdat)
	if (xlog) pdat = log10(tdat + tmin)
	else pdat = tdat
        
        for (i in rev(2:nrow(tdat))) {
            tdat[i,] = tdat[i,] - tdat[i-1,]
            #browser()
        }
        #browser()
        if (deg_j > 1) tdat = tdat - tdat0
        else tdat0 <<- tdat
	plot_lines <- function(i) {
		plot_line <- function(j) {
			lines(c(pdat[i-1, j], pdat[i, j]), yl -0.5 + j/ncol(pdat), 
				  lwd = 6, col = 'black')
			lines(c(pdat[i-1, j], pdat[i, j]), yl -0.5 + j/ncol(pdat), 
				  lwd = 5, col = cols[i-1])
			
		}
                
		if (i > 1) lapply(1:ncol(pdat), plot_line)
		#if (i == nrow(pdat)) return()
		#if (deg_j > 1) 
		    txt = round(tdat[i,], 2) #  - dat[[1]][[1]][i-1,]
		#else 
		#    txt = round(tdat[i,], 2)
		days_to_time <- function(x) {   
                    if (x<0) neg = TRUE else neg = FALSE
                    x = abs(x * 24)
                    if (round(x) == 0 ) return(x)
                    weeks = floor(x/(7*24))
                    days = floor(x/24 - 7*weeks)
                    hours = round(x - 24*days - 24*7*weeks)
                    if (hours > 24) browser()
                    out = paste0(hours, 'hrs')
                    if (days > 0 || weeks > 0) out = paste0(days, 'days, ', out)
                    if ( weeks > 0) out = paste0(weeks, 'weeks, ', out)
                    if (neg) out = paste0('-', out)
                    return(out)
                }
                txt0 = txt
		if (!any(is.na(txt)) && !all(txt == 0)) {
                    txt = sort(txt)                    
                    #txt = sapply(txt, days_to_time)
		    if (txt[1] != txt[3]) txt = paste0(txt[2], '\n(', txt[1], '-', txt[3], ')')
                    else txt = txt[2]
                } else {
                    txt = 0
                }
                if ((i/2) == round(i/2)) yd = 1.2 else yd = -0.9
                #if (txt == '') browser()
		if (txt != 0)
                    text(mean(c(pdat[i,2], pdat[i-1,2])), yl + yd, txt, cex = 0.6, xpd = TRUE)
                txt = gsub('\n', ' ', txt)
                return(txt)
	}
	outs = sapply(2:nrow(pdat), plot_lines)
        
}

png("time_in_ffdi_range2_all.png", height = 14, width = 9, units = 'in', res = 300)
layout(rbind(1:2, 3:4, 5), heights = c(1, 1, 0.1))
par(mar = c(0, 3, 0.2, 2.2), oma = c(3, 1, 0, 0))

plot_region <- function(dat, name) {
    xrange = c(0, max(unlist(dat, recursive = TRUE), na.rm = TRUE))
    xrange = c(0, 380)
    if (xlog) xrange = log10(xrange + tmin)
    plot(xrange, 1.43*c(0, length(degs) * (2.5+length(rcps))), axes = FALSE,
	 type = 'n', xlab = '', ylab = '', xaxs = 'i')
    mtext(side = 3, line = -4.5, name, font = 2)
    plot_Deg <- function(j) {
        outs = lapply(1:length(rcps), plot_rcpDeg, j, dat)
        outs[sapply(outs, is.null)][[1]] = rep('', 6)
        do.call(rbind, outs)
    }
    outs = lapply(1:length(degs), plot_Deg)   	
    outs = do.call(rbind,outs)  
    outs = outs[c(2:8, 10), ]
    rownames(outs) = c('Baseline 1986-2005', 
                       unlist(lapply(c('2.6', '8.5'), 
                                     function(rcp) paste0('RCP', rcp,' ', 
                                                          c('Recent 2004-2023', 
                                                            'GWL1.5', 'GWL2')))), 
                       'RCP8.5 GWL4')
    
    colnames(outs) = labs
    write.csv(outs,  paste0('outputs/FWI_catogory-', name, '.csv'))
    if (xlog) {
    	at = c(0, 1/24, 1/4, 1, 7, 30, 365)
    	at = log10(at + tmin)
	labels = c(0, 'hour', '6-hr', 'day', 'week', '~month', 'year')
	axis(1, at = at, labels = labels)
    } else axis(1)

    text(x = xrange[1], y = 1.5*0.5, '1986-\n2005', 
	 srt = 90, xpd = NA, adj = c(0.5, -1))
    text(x = xrange[1], y = 1.5*4.5, '\u0394 2004-\n2023', 
	 srt = 90, xpd = NA, adj = c(0.5, -1))
    text(x = xrange[1], 
	 y =  1.5*(4.5+ (length(rcps) + 2.5) * (1:(length(degs)-2))), 
	 paste0('\u0394 GWL', degs[-(1:2)], '\u00B0C'), 
	 srt = 90, xpd = TRUE, adj = c(0.5, -2.75))
    for (i in 1:length(rcps)) {
	text(x = xrange[1], 
		 y = 1.5*(i*2 - 3 + (length(rcps) + 2.5) * (1:(length(degs)-1))), 
	     names(rcps)[i], 
	 srt = 90, xpd = TRUE, adj = c(0.5, -1.25))
    }
    return(outs)
#axis(2, at = 0.5+ (length(rcps) + 1) * (1:(length(degs)-1)), labels = degs[-1])
    
}
outs =mapply(plot_region,dats, names(country))
plot(c(-0.5, 1.5), c(0, 11), xlab ='', ylab = '', axes = FALSE, type = 'n')
addLine <- function(col, x, lab, y) {
    y = rep(y, 2)
    xp = c(x-1, x)/length(cols)
    lines(xp, y, lwd = 6, col = 'black', xpd = NA)
    lines(xp, y, lwd = 5, col = col, xpd = NA)
    if (y==3) {
        if (round(x/2) == x/2) yp = -0.5 else yp = 4.5
        text(mean(xp), yp, lab, xpd = NA)
    }   
        
}
labs = rev(c("Low–moderate", "High", "Very High", "Severe", "Extreme", "Catastrophic"))
lapply(1:3, function(y) mapply(addLine, cols, 1:length(cols), labs, y))
text(-0.02, 0.5, adj = 1, '90th percentile', xpd = NA, cex = 0.8)
text(-0.02, 2, adj = 1, 'mean', xpd = NA, cex = 0.8)
text(-0.02, 3.5, adj = 1, '10th percentile', xpd = NA, cex = 0.8)
dev.off()



